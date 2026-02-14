"""
nanollama.py — An educational LLM inference engine in ~1000 lines of PyTorch.

Loads a LLaMA/Qwen2-architecture model from HuggingFace and generates text.
Every component of the transformer is implemented from scratch so you
can see exactly how LLMs work at the tensor level.

Usage:
    python nanollama.py --prompt "Once upon a time"
    python nanollama.py --prompt "What is 2+2?" --chat --dtype float16
    python nanollama.py --serve --dtype float16 --port 8000

First run downloads the model from HuggingFace (~3GB). Subsequent runs use the cache.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer


# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Model hyperparameters, read from HuggingFace config.json."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 22
    num_attention_heads: int = 32       # query heads
    num_key_value_heads: int = 4        # key/value heads (fewer = GQA)
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_bias: bool = False         # Qwen2 uses bias on QKV projections

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        with open(path) as f:
            d = json.load(f)
        fields = {f.name for f in __import__("dataclasses").fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ── RMSNorm ───────────────────────────────────────────────────────────────
# Like LayerNorm but without mean-centering. Simpler and works just as well.
# Formula: x * weight / sqrt(mean(x²) + eps)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ── Quantization ─────────────────────────────────────────────────────────
# Store weights as low-bit integers + a scale factor to save memory.
# On forward pass, dequantize back to float for the matmul.
# Q8 (int8): ~4x memory savings. Q4 (4-bit packed into int8): ~8x savings.
# Per-channel (absmax) quantization: scale = max(|w|) / max_int per output row.

class QuantizedLinear(nn.Module):
    """Drop-in nn.Linear replacement with quantized weights."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, bits: int = 8):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.bits = bits
        max_int = (1 << (bits - 1)) - 1  # 127 for Q8, 7 for Q4

        # Per-output-channel absmax scale
        scale = weight.float().abs().amax(dim=1) / max_int  # [out_features]
        quantized = (weight.float() / scale.unsqueeze(1)).round().clamp(-max_int - 1, max_int).to(torch.int8)

        if bits == 4:
            # Pack two 4-bit values into one int8: high nibble + low nibble
            # Pad columns to even count if needed
            if quantized.shape[1] % 2 != 0:
                quantized = F.pad(quantized, (0, 1))
            even, odd = quantized[:, 0::2], quantized[:, 1::2]
            quantized = (even << 4) | (odd & 0x0F)

        self.register_buffer("qweight", quantized)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        if self.bits == 4:
            # Unpack: extract high nibble (arithmetic shift preserves sign) and low nibble
            w = self.qweight
            high = w >> 4                        # sign-extending arithmetic shift
            low = (w << 4).to(torch.int8) >> 4   # shift left then arithmetic shift right
            unpacked = torch.stack([high, low], dim=-1).reshape(self.out_features, -1)
            unpacked = unpacked[:, :self.in_features]  # trim padding
            w_deq = unpacked.float() * self.scale.unsqueeze(1)
        else:
            w_deq = self.qweight.float() * self.scale.unsqueeze(1)
        out = x.float() @ w_deq.T
        if self.bias is not None:
            out = out + self.bias.float()
        return out.type_as(x)


def quantize_model(model: nn.Module, bits: int = 8):
    """Replace all nn.Linear layers with QuantizedLinear (in-place)."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear(child.weight.data, child.bias.data if child.bias is not None else None, bits))
        else:
            quantize_model(child, bits)


# ── Rotary Positional Embeddings (RoPE) ───────────────────────────────────
# Encodes position by rotating pairs of dimensions. The dot product of two
# rotated vectors depends only on their RELATIVE position — this gives the
# model translation-invariant position awareness.
# LLaMA uses "split-half" pairing: dimension pairs are (d, d+dim//2) rather
# than interleaved (d, d+1). The rotation is: [x1*cos - x2*sin, x2*cos + x1*sin]
# where x1 is the first half of dimensions and x2 is the second half.

def precompute_rope(dim: int, max_seq: int, theta: float = 10000.0):
    """Build cos/sin tables. Each dimension-pair gets a different frequency."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.outer(torch.arange(max_seq).float(), freqs)
    return angles.cos(), angles.sin()


def apply_rope(q, k, cos, sin, pos):
    """Rotate q and k by position-dependent angles.
    pos: int (single offset for all seqs) or tensor [B, S] (per-token positions).
    """
    if isinstance(pos, int):
        seq = q.shape[1]
        cos = cos[pos:pos + seq].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim//2]
        sin = sin[pos:pos + seq].unsqueeze(0).unsqueeze(2)
    else:
        # Batched position_ids [B, S] — gather per-token cos/sin
        cos = cos[pos].unsqueeze(2)  # [B, S, 1, dim//2]
        sin = sin[pos].unsqueeze(2)

    def rotate(x):
        x = x.float()
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin,
                          x2 * cos + x1 * sin], dim=-1).type_as(q)

    return rotate(q), rotate(k)


# ── Grouped-Query Attention with KV-Cache ─────────────────────────────────
# GQA: multiple query heads share one key/value head (saves memory).
# KV-Cache: stores past keys/values so we only process the new token each step.

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv  # how many Q heads per KV head

        qkv_bias = cfg.attention_bias  # LLaMA: False, Qwen2: True
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.cache_k = self.cache_v = None

    def forward(self, x, cos, sin, pos, mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv, self.head_dim)
        q, k = apply_rope(q, k, cos, sin, pos)

        # Update KV cache
        if self.cache_k is None:
            self.cache_k = torch.zeros(B, cos.shape[0], self.n_kv, self.head_dim,
                                       dtype=x.dtype, device=x.device)
            self.cache_v = torch.zeros_like(self.cache_k)
        if isinstance(pos, int):
            self.cache_k[:B, pos:pos + S] = k
            self.cache_v[:B, pos:pos + S] = v
            kv_len = pos + S
        else:
            # Batched position_ids [B, S] — write real tokens to cache positions.
            # Left-padded inputs have duplicate position 0s (padding + first real token).
            # Loop ensures the real token (rightmost) wins deterministically.
            for b in range(B):
                for s in range(S):
                    p = pos[b, s].item()
                    self.cache_k[b, p] = k[b, s]
                    self.cache_v[b, p] = v[b, s]
            kv_len = pos.max().item() + 1
        k = self.cache_k[:B, :kv_len]
        v = self.cache_v[:B, :kv_len]

        # Expand KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            k = k.reshape(B, -1, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            v = v.reshape(B, -1, self.n_heads, self.head_dim)

        # Scaled dot-product attention (scores computed in float32 for stability
        # with float16, where MPS accumulates in half-precision unlike CUDA)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask
        attn = F.softmax(scores, dim=-1).type_as(q)
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, S, -1)
        return self.o_proj(out)


# ── SwiGLU Feed-Forward Network ──────────────────────────────────────────
# Gated FFN: output = down(silu(gate(x)) * up(x))
# The gate controls which features pass through — more expressive than ReLU FFN.

class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer Block ────────────────────────────────────────────────────
# Pre-norm residual: x → norm → sublayer → + residual
# Repeating this block N times is what makes the model deep.

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = FFN(cfg)

    def forward(self, x, cos, sin, pos, mask=None):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, pos, mask)
        return h + self.mlp(self.post_attention_layernorm(h))


# ── Full Transformer ─────────────────────────────────────────────────────
# token_ids → embedding → N blocks → norm → logits (next-token scores)

class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.max_seq_len = cfg.max_position_embeddings
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        hd = cfg.hidden_size // cfg.num_attention_heads
        self.rope_cos, self.rope_sin = precompute_rope(
            hd, cfg.max_position_embeddings, cfg.rope_theta
        )

    def forward(self, tokens, start_pos=0, position_ids=None, pad_mask=None):
        """Forward pass.
        Args:
            tokens: [B, S] token IDs
            start_pos: int position offset (used when position_ids is None)
            position_ids: [B, S] per-token position indices for RoPE (for batched/padded inputs)
            pad_mask: [B, total_seq_len] bool — True for real tokens, False for padding
        """
        h = self.embed_tokens(tokens)
        cos = self.rope_cos.to(h.device)
        sin = self.rope_sin.to(h.device)

        # Use position_ids for RoPE if provided, else fall back to start_pos
        pos = position_ids if position_ids is not None else start_pos

        # Causal mask: prevent attending to future positions
        mask = None
        B, seq = tokens.shape
        kv_len = (position_ids.max().item() + 1) if position_ids is not None else start_pos + seq
        if seq > 1:
            mask = torch.full((seq, seq), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            prefix = kv_len - seq  # cached positions before current tokens
            if prefix > 0:
                mask = torch.cat([torch.zeros(seq, prefix, device=h.device), mask], -1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, kv_len]

        # Padding mask: block attention to padding positions in the KV cache
        if pad_mask is not None:
            # Use torch.where to avoid 0.0 * -inf = NaN (IEEE 754)
            pad_attn = torch.where(
                pad_mask[:, :kv_len], 0.0, float("-inf")
            ).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, kv_len]
            mask = pad_attn if mask is None else mask + pad_attn

        for layer in self.layers:
            h = layer(h, cos, sin, pos, mask)

        return self.lm_head(self.norm(h))

    def reset(self):
        for layer in self.layers:
            layer.self_attn.cache_k = layer.self_attn.cache_v = None


# ── Loading ───────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Download a model from HuggingFace and load it. Returns (model, tokenizer)."""
    path = Path(model_id)
    if not path.exists():
        print(f"Downloading {model_id}...")
        path = Path(snapshot_download(model_id, ignore_patterns=["*.md", "*.txt"]))

    cfg = Config.from_json(path / "config.json")

    # Load weights first so we can detect architecture features
    weights = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(load_file(f))
    # HuggingFace prefixes weights with "model." — strip it
    mapped = {(k[6:] if k.startswith("model.") else k): v for k, v in weights.items()}
    # Some models tie lm_head to embed_tokens (no separate lm_head in weights)
    if "lm_head.weight" not in mapped:
        mapped["lm_head.weight"] = mapped["embed_tokens.weight"]

    # Auto-detect QKV bias from weights (Qwen2 uses bias, LLaMA doesn't)
    if "layers.0.self_attn.q_proj.bias" in mapped:
        cfg.attention_bias = True

    # Build model and load weights
    model = Transformer(cfg)
    model.load_state_dict(mapped, strict=False)
    model.to(dtype=dtype, device=device).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(path))

    params = sum(p.numel() for p in model.parameters())
    dtype_str = {torch.float32: "float32", torch.float16: "float16",
                 torch.bfloat16: "bfloat16"}.get(dtype, str(dtype))
    print(f"Loaded {cfg.num_hidden_layers} layers, {params / 1e9:.1f}B params on {device} ({dtype_str})")
    return model, tokenizer


# ── Chat Template ─────────────────────────────────────────────────────────
# Chat-tuned models are fine-tuned on text with special role markers like
# <|user|> and <|assistant|>. Without these markers, the model doesn't know
# it should "respond" — it just continues the text like autocomplete.
# Each model stores its template as a Jinja2 string in tokenizer_config.json.
# We load and render it so any model's chat format works automatically —
# this is how HuggingFace and Ollama support many formats with one codebase.

CHATML_FALLBACK = "<|user|>\n{prompt}</s>\n<|assistant|>\n"


def apply_chat_template(messages: list[dict], tokenizer) -> str:
    """Format messages using the model's Jinja2 chat template.

    Args:
        messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
        tokenizer: HuggingFace tokenizer (provides chat_template and special tokens)
    """
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        from jinja2 import BaseLoader, Environment
        env = Environment(loader=BaseLoader(), keep_trailing_newline=True,
                          trim_blocks=True, lstrip_blocks=True)
        template = env.from_string(tokenizer.chat_template)
        return template.render(
            messages=messages, add_generation_prompt=True,
            bos_token=tokenizer.bos_token or "",
            eos_token=tokenizer.eos_token or "",
        )
    # Fallback: render last user message in ChatML format
    last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    return CHATML_FALLBACK.format(prompt=last_user)


# ── Generation ────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_tokens: int = 200,
             temperature: float = 0.7, top_k: int = 0, top_p: float = 1.0,
             repeat_penalty: float = 1.0, add_bos: bool = True):
    """Generate text from a prompt. Streams tokens to stdout."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if add_bos:
        ids = [tokenizer.bos_token_id] + ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    model.reset()

    # Prefill: process entire prompt at once
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tokens, start_pos=0)
    t_prefill = time.perf_counter() - t0

    def sample(logits, token_history):
        logits = logits.clone()
        # Repetition penalty: discourage tokens that already appeared.
        # Positive logits are divided by the penalty (reduced probability),
        # negative logits are multiplied (made more negative). This always
        # reduces the chance of repeating a token, regardless of sign.
        if repeat_penalty != 1.0 and token_history:
            for tid in set(token_history):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repeat_penalty
                else:
                    logits[0, tid] *= repeat_penalty
        if temperature == 0:
            return logits.argmax(-1).item()
        logits = logits / temperature
        # Top-k: keep only the k highest logits, set the rest to -inf.
        # This removes the long tail of unlikely tokens before sampling,
        # preventing the model from occasionally picking nonsense tokens.
        if top_k > 0:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[:, -1, None]] = float("-inf")
        # Top-p (nucleus): keep the smallest set of tokens whose cumulative
        # probability exceeds p. Unlike top-k which always keeps a fixed count,
        # top-p adapts — fewer tokens when confident, more when uncertain.
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Remove tokens whose cumulative probability is above the threshold,
            # but always keep at least one token (shift right by 1)
            remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.zeros_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), 1).item()

    # First generated token
    next_id = sample(logits[:, -1], ids)
    generated = [next_id]
    prev_text = ""
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text, end="", flush=True)
    prev_text = text

    # Decode: generate one token at a time using KV cache
    t_dec = time.perf_counter()
    for i in range(1, max_tokens):
        if next_id == tokenizer.eos_token_id:
            break
        inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(inp, start_pos=len(ids) + i - 1)
        next_id = sample(logits[:, -1], ids + generated)
        generated.append(next_id)
        # Decode full sequence to get correct spacing from the tokenizer
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(text[len(prev_text):], end="", flush=True)
        prev_text = text
    t_dec_total = time.perf_counter() - t_dec

    n_gen = len(generated)
    print(f"\n\n[prefill: {len(ids)} tok @ {len(ids) / t_prefill:.0f} t/s"
          f" | decode: {n_gen} tok @ {max(1, n_gen - 1) / t_dec_total:.1f} t/s]")
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_streaming(model, tokenizer, prompt: str, max_tokens: int = 200,
                       temperature: float = 0.7, top_k: int = 0, top_p: float = 1.0,
                       repeat_penalty: float = 1.0, add_bos: bool = True):
    """Generate text, yielding each new text fragment as it's produced.

    Yields dicts: {"text": delta_text, "token_id": int, "finish_reason": None|"stop"|"length"}
    Same prefill + decode loop as generate(), but yields instead of printing.
    """
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if add_bos:
        ids = [tokenizer.bos_token_id] + ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    model.reset()

    # Prefill: process entire prompt at once
    with torch.no_grad():
        logits = model(tokens, start_pos=0)

    def sample(logits, token_history):
        logits = logits.clone()
        if repeat_penalty != 1.0 and token_history:
            for tid in set(token_history):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repeat_penalty
                else:
                    logits[0, tid] *= repeat_penalty
        if temperature == 0:
            return logits.argmax(-1).item()
        logits = logits / temperature
        if top_k > 0:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[:, -1, None]] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.zeros_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), 1).item()

    # First generated token
    next_id = sample(logits[:, -1], ids)
    generated = [next_id]
    prev_text = ""
    text = tokenizer.decode(generated, skip_special_tokens=True)
    delta = text
    prev_text = text

    if next_id == tokenizer.eos_token_id:
        yield {"text": delta, "token_id": next_id, "finish_reason": "stop"}
        return
    yield {"text": delta, "token_id": next_id, "finish_reason": None}

    # Decode: generate one token at a time using KV cache
    for i in range(1, max_tokens):
        inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(inp, start_pos=len(ids) + i - 1)
        next_id = sample(logits[:, -1], ids + generated)
        generated.append(next_id)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        delta = text[len(prev_text):]
        prev_text = text

        if next_id == tokenizer.eos_token_id:
            yield {"text": delta, "token_id": next_id, "finish_reason": "stop"}
            return
        if i == max_tokens - 1:
            yield {"text": delta, "token_id": next_id, "finish_reason": "length"}
            return
        yield {"text": delta, "token_id": next_id, "finish_reason": None}


# ── API Server ─────────────────────────────────────────────────────────────
# OpenAI-compatible HTTP API so external tools (LangChain, Open WebUI, curl)
# can use nanollama as a drop-in local LLM backend.
# An asyncio.Lock serializes requests — the model has mutable KV-cache state
# that can't be shared across concurrent generations.

def create_app(model, tokenizer):
    """Create a FastAPI app with OpenAI-compatible endpoints."""
    import asyncio
    import uuid

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="nanollama", description="OpenAI-compatible API for nanollama")
    model_lock = asyncio.Lock()

    # Determine the model name from the tokenizer (or fall back to a generic name)
    model_name = getattr(tokenizer, "name_or_path", "nanollama")

    # ── Request/Response Schemas ──────────────────────────────────────────

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        messages: list[ChatMessage]
        model: str | None = None
        temperature: float = 0.7
        max_tokens: int = 200
        top_p: float = 0.9
        top_k: int = 50
        repeat_penalty: float = 1.1
        stream: bool = False

    class CompletionRequest(BaseModel):
        prompt: str
        model: str | None = None
        temperature: float = 0.7
        max_tokens: int = 200
        top_p: float = 0.9
        top_k: int = 50
        repeat_penalty: float = 1.1
        stream: bool = False

    # ── Endpoints ─────────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "owned_by": "nanollama",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = apply_chat_template(messages, tokenizer)
        sampling = dict(
            max_tokens=request.max_tokens, temperature=request.temperature,
            top_k=request.top_k, top_p=request.top_p,
            repeat_penalty=request.repeat_penalty, add_bos=False,
        )

        if request.stream:
            return StreamingResponse(
                _stream_chat(prompt, sampling, request.model or model_name),
                media_type="text/event-stream",
            )

        async with model_lock:
            result = await asyncio.to_thread(
                generate, model, tokenizer, prompt, **sampling
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        completion_tokens = len(tokenizer.encode(result, add_special_tokens=False))
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        sampling = dict(
            max_tokens=request.max_tokens, temperature=request.temperature,
            top_k=request.top_k, top_p=request.top_p,
            repeat_penalty=request.repeat_penalty, add_bos=True,
        )

        if request.stream:
            return StreamingResponse(
                _stream_completion(request.prompt, sampling, request.model or model_name),
                media_type="text/event-stream",
            )

        async with model_lock:
            result = await asyncio.to_thread(
                generate, model, tokenizer, request.prompt, **sampling
            )

        completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        prompt_tokens = len(tokenizer.encode(request.prompt, add_special_tokens=False))
        completion_tokens = len(tokenizer.encode(result, add_special_tokens=False))
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model or model_name,
            "choices": [{
                "index": 0,
                "text": result,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # ── SSE Streaming Helpers ─────────────────────────────────────────────
    # OpenAI SSE format: each chunk is "data: {json}\n\n", ending with "data: [DONE]\n\n"

    async def _stream_chat(prompt, sampling, req_model):
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        async with model_lock:
            for chunk in await asyncio.to_thread(
                lambda: list(generate_streaming(model, tokenizer, prompt, **sampling))
            ):
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk["text"]},
                        "finish_reason": chunk["finish_reason"],
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_completion(prompt, sampling, req_model):
        completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        async with model_lock:
            for chunk in await asyncio.to_thread(
                lambda: list(generate_streaming(model, tokenizer, prompt, **sampling))
            ):
                data = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": req_model,
                    "choices": [{
                        "index": 0,
                        "text": chunk["text"],
                        "finish_reason": chunk["finish_reason"],
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    return app


# ── Batched Generation ────────────────────────────────────────────────────
# Process multiple prompts in a single forward pass for higher throughput.
# Left-pad shorter prompts so all sequences align on the right (the generation
# side). position_ids track the real position of each token for RoPE, and a
# padding mask prevents attention to padding positions.

def generate_batch(model, tokenizer, prompts: list[str], max_tokens: int = 200,
                   temperature: float = 0.7, top_k: int = 0, top_p: float = 1.0,
                   repeat_penalty: float = 1.0, add_bos: bool = True):
    """Generate text from multiple prompts simultaneously."""
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # Tokenize all prompts
    all_ids = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if add_bos:
            ids = [tokenizer.bos_token_id] + ids
        all_ids.append(ids)

    # Left-pad to same length
    max_len = max(len(ids) for ids in all_ids)
    padded = []
    pad_lengths = []
    for ids in all_ids:
        pad_len = max_len - len(ids)
        pad_lengths.append(pad_len)
        padded.append([pad_id] * pad_len + ids)

    B = len(prompts)
    tokens = torch.tensor(padded, dtype=torch.long, device=device)  # [B, max_len]

    # Position IDs: real tokens get 0, 1, 2, ...; padding gets 0 (masked out anyway)
    position_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, pl in enumerate(pad_lengths):
        position_ids[i, pl:] = torch.arange(max_len - pl, device=device)

    # Padding mask for CACHE positions (not input positions).
    # With left-padding, cache positions 0..(real_len-1) have real content.
    # Position real_len..(max_len-1) may be empty for shorter sequences.
    pad_mask = torch.zeros(B, max_len + max_tokens, dtype=torch.bool, device=device)
    for i, pl in enumerate(pad_lengths):
        real_len = max_len - pl
        pad_mask[i, :real_len] = True

    model.reset()

    # Prefill: process all prompts in one batched forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tokens, position_ids=position_ids, pad_mask=pad_mask)
    t_prefill = time.perf_counter() - t0

    # Sample first token for each sequence
    generated = [[] for _ in range(B)]
    finished = [False] * B
    next_ids = []
    for i in range(B):
        row_logits = logits[i:i+1, -1]
        row_logits = row_logits.clone()
        if temperature == 0:
            nid = row_logits.argmax(-1).item()
        else:
            row_logits = row_logits / temperature
            if top_k > 0:
                top_values, _ = torch.topk(row_logits, min(top_k, row_logits.size(-1)))
                row_logits[row_logits < top_values[:, -1, None]] = float("-inf")
            nid = torch.multinomial(F.softmax(row_logits, dim=-1), 1).item()
        generated[i].append(nid)
        next_ids.append(nid)
        if nid == eos_id:
            finished[i] = True

    # Decode: generate one token at a time, all sequences in lockstep.
    # Each sequence tracks its own cache position (real_len + decode_step)
    # since shorter sequences started at lower positions in the cache.
    t_dec = time.perf_counter()
    cur_positions = [max_len - pl for pl in pad_lengths]  # next write position per sequence
    for step in range(1, max_tokens):
        if all(finished):
            break
        inp = torch.tensor([[nid] for nid in next_ids], dtype=torch.long, device=device)
        pos_ids = torch.tensor([[p] for p in cur_positions], dtype=torch.long, device=device)
        # Mark new cache positions as valid in the pad_mask
        for i in range(B):
            pad_mask[i, cur_positions[i]] = True
        with torch.no_grad():
            logits = model(inp, position_ids=pos_ids, pad_mask=pad_mask)
        cur_positions = [p + 1 for p in cur_positions]

        for i in range(B):
            if finished[i]:
                next_ids[i] = pad_id
                continue
            row_logits = logits[i:i+1, -1].clone()
            history = all_ids[i] + generated[i]
            if repeat_penalty != 1.0 and history:
                for tid in set(history):
                    if row_logits[0, tid] > 0:
                        row_logits[0, tid] /= repeat_penalty
                    else:
                        row_logits[0, tid] *= repeat_penalty
            if temperature == 0:
                nid = row_logits.argmax(-1).item()
            else:
                row_logits = row_logits / temperature
                if top_k > 0:
                    top_values, _ = torch.topk(row_logits, min(top_k, row_logits.size(-1)))
                    row_logits[row_logits < top_values[:, -1, None]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(row_logits, descending=True)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[remove] = float("-inf")
                    row_logits = torch.zeros_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)
                nid = torch.multinomial(F.softmax(row_logits, dim=-1), 1).item()
            generated[i].append(nid)
            next_ids[i] = nid
            if nid == eos_id:
                finished[i] = True
    t_dec_total = time.perf_counter() - t_dec

    results = []
    for i in range(B):
        text = tokenizer.decode(generated[i], skip_special_tokens=True)
        results.append(text)

    total_prompt = sum(len(ids) for ids in all_ids)
    total_gen = sum(len(g) for g in generated)
    print(f"\n[batch={B} | prefill: {total_prompt} tok @ {total_prompt / t_prefill:.0f} t/s"
          f" | decode: {total_gen} tok @ {max(1, total_gen - B) / t_dec_total:.1f} t/s]")
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nanollama — educational LLM inference")
    parser.add_argument("--prompt", default=None, help="input prompt")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps (auto-detected)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                        help="model precision: float32, float16, bfloat16 (default: float32)")
    parser.add_argument("--temp", type=float, default=0.7, help="sampling temperature (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=50, help="top-k filtering: keep k most likely tokens, 0=disabled (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.9, help="top-p nucleus sampling threshold, 1.0=disabled (default: 0.9)")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="repetition penalty: 1.0=off, 1.1=mild, 1.3=strong (default: 1.1)")
    parser.add_argument("--max-tokens", type=int, default=200, help="max tokens to generate (default: 200)")
    parser.add_argument("--chat", action="store_true", help="wrap prompt in ChatML template for chat-tuned models")
    parser.add_argument("--interactive", action="store_true", help="interactive REPL mode (implies --chat)")
    parser.add_argument("--system", default=None, help="system prompt to steer model behavior (used with --chat or --interactive)")
    parser.add_argument("--quantize", default=None, choices=["q8", "q4"],
                        help="post-load weight quantization: q8 (int8, ~4x savings) or q4 (4-bit, ~8x savings)")
    parser.add_argument("--compile", action="store_true",
                        help="use torch.compile() for kernel fusion (slower first run, faster generation)")
    parser.add_argument("--batch-file", default=None,
                        help="file with one prompt per line for batched generation")
    parser.add_argument("--serve", action="store_true",
                        help="start OpenAI-compatible API server instead of generating")
    parser.add_argument("--port", type=int, default=8000,
                        help="server port (default: 8000, used with --serve)")
    args = parser.parse_args()

    if not args.interactive and not args.prompt and not args.batch_file and not args.serve:
        parser.error("--prompt is required (unless using --interactive, --batch-file, or --serve)")

    device = args.device or auto_device()
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    model, tokenizer = load_model(args.model, device, dtype)

    if args.quantize:
        bits = {"q8": 8, "q4": 4}[args.quantize]
        print(f"Quantizing to {args.quantize.upper()}...")
        quantize_model(model, bits)
        qparams = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
        qbytes = sum(b.nbytes for b in model.buffers()) + sum(p.nbytes for p in model.parameters())
        print(f"Quantized: {qbytes / 1e9:.1f}GB ({args.quantize.upper()})")

    if args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    sampling = dict(max_tokens=args.max_tokens, temperature=args.temp,
                    top_k=args.top_k, top_p=args.top_p,
                    repeat_penalty=args.repeat_penalty)

    if args.serve:
        import uvicorn
        app = create_app(model, tokenizer)
        print(f"\nStarting server on http://0.0.0.0:{args.port}")
        print(f"OpenAI-compatible API: http://localhost:{args.port}/v1/chat/completions")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.interactive:
        # REPL loop with multi-turn history. Implies --chat.
        # Each turn appends to the conversation and re-renders the full
        # template so the model sees the entire context. If the conversation
        # exceeds the context window, the oldest turns are dropped.
        print(f"\nDevice: {device}")
        print("Interactive mode — type a message, press Enter to generate.")
        print("Press Ctrl+C or type /exit to quit.\n")
        history: list[dict] = []
        if args.system:
            history.append({"role": "system", "content": args.system})
        try:
            while True:
                try:
                    user_input = input("> ")
                except EOFError:
                    break
                if not user_input.strip():
                    continue
                if user_input.strip() == "/exit":
                    break
                history.append({"role": "user", "content": user_input})
                prompt = apply_chat_template(history, tokenizer)
                # Truncate oldest turns if prompt exceeds context window
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                while len(ids) > model.max_seq_len - sampling["max_tokens"] and len(history) > 1:
                    history.pop(0)
                    prompt = apply_chat_template(history, tokenizer)
                    ids = tokenizer.encode(prompt, add_special_tokens=False)
                reply = generate(model, tokenizer, prompt, add_bos=False, **sampling)
                history.append({"role": "assistant", "content": reply})
                print()
        except KeyboardInterrupt:
            print()
    elif args.batch_file:
        # Batched generation: process multiple prompts simultaneously
        with open(args.batch_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Device: {device}")
        print(f"Batch: {len(prompts)} prompts\n")
        if args.chat:
            prompts = [apply_chat_template([{"role": "user", "content": p}], tokenizer) for p in prompts]
            add_bos = False
        else:
            add_bos = True
        results = generate_batch(model, tokenizer, prompts, add_bos=add_bos, **sampling)
        for i, (p, r) in enumerate(zip(prompts, results)):
            print(f"\n--- Prompt {i + 1} ---")
            print(r)
    else:
        print(f"Device: {device}")
        print(f"Prompt: {args.prompt}\n")
        if args.chat:
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": args.prompt})
            prompt = apply_chat_template(messages, tokenizer)
            add_bos = False
        else:
            prompt = args.prompt
            add_bos = True
        print()
        generate(model, tokenizer, prompt, add_bos=add_bos, **sampling)
