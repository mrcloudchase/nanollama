"""
nanollama.py — An educational LLM inference engine in ~300 lines of PyTorch.

Loads a LLaMA-architecture model from HuggingFace and generates text.
Every component of the transformer is implemented from scratch so you
can see exactly how LLMs work at the tensor level.

Usage:
    python nanollama.py --prompt "Once upon a time"
    python nanollama.py --prompt "What is 2+2?" --temp 0

First run downloads TinyLlama (~2GB). Subsequent runs use the cache.
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


# ── Rotary Positional Embeddings (RoPE) ───────────────────────────────────
# Encodes position by rotating pairs of dimensions. The dot product of two
# rotated vectors depends only on their RELATIVE position — this gives the
# model translation-invariant position awareness.

def precompute_rope(dim: int, max_seq: int, theta: float = 10000.0):
    """Build cos/sin tables. Each dimension-pair gets a different frequency."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.outer(torch.arange(max_seq).float(), freqs)
    return angles.cos(), angles.sin()


def apply_rope(q, k, cos, sin, pos):
    """Rotate q and k by position-dependent angles."""
    seq = q.shape[1]
    cos = cos[pos:pos + seq].unsqueeze(0).unsqueeze(2)
    sin = sin[pos:pos + seq].unsqueeze(0).unsqueeze(2)

    def rotate(x):
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        out = torch.stack([even * cos - odd * sin,
                           even * sin + odd * cos], dim=-1)
        return out.reshape(x.shape).type_as(x)

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

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
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
        self.cache_k[:B, pos:pos + S] = k
        self.cache_v[:B, pos:pos + S] = v
        k = self.cache_k[:B, :pos + S]
        v = self.cache_v[:B, :pos + S]

        # Expand KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            k = k.reshape(B, -1, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            v = v.reshape(B, -1, self.n_heads, self.head_dim)

        # Scaled dot-product attention
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask
        attn = F.softmax(scores.float(), dim=-1).type_as(q)
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
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        hd = cfg.hidden_size // cfg.num_attention_heads
        self.rope_cos, self.rope_sin = precompute_rope(
            hd, cfg.max_position_embeddings, cfg.rope_theta
        )

    def forward(self, tokens, start_pos=0):
        h = self.embed_tokens(tokens)
        cos = self.rope_cos.to(h.device)
        sin = self.rope_sin.to(h.device)

        # Causal mask: prevent attending to future positions
        mask = None
        seq = tokens.shape[1]
        if seq > 1:
            mask = torch.full((seq, seq), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            if start_pos > 0:
                mask = torch.cat([torch.zeros(seq, start_pos, device=h.device), mask], -1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, cos, sin, start_pos, mask)

        return self.lm_head(self.norm(h))

    def reset(self):
        for layer in self.layers:
            layer.self_attn.cache_k = layer.self_attn.cache_v = None


# ── Loading ───────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str = "cpu"):
    """Download a model from HuggingFace and load it. Returns (model, tokenizer)."""
    path = Path(model_id)
    if not path.exists():
        print(f"Downloading {model_id}...")
        path = Path(snapshot_download(model_id, ignore_patterns=["*.md", "*.txt"]))

    cfg = Config.from_json(path / "config.json")

    # Build model and load weights
    model = Transformer(cfg)
    weights = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(load_file(f))
    # HuggingFace prefixes weights with "model." — strip it
    mapped = {(k[6:] if k.startswith("model.") else k): v for k, v in weights.items()}
    model.load_state_dict(mapped, strict=False)
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(path))

    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {cfg.num_hidden_layers} layers, {params / 1e9:.1f}B params on {device}")
    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_tokens: int = 200,
             temperature: float = 0.7, top_k: int = 0, top_p: float = 1.0,
             repeat_penalty: float = 1.0):
    """Generate text from a prompt. Streams tokens to stdout."""
    device = next(model.parameters()).device
    ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    model.reset()

    # Prefill: process entire prompt at once
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tokens, start_pos=0)
    t_prefill = time.perf_counter() - t0

    def sample(logits, token_history):
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
            logits = sorted_logits.scatter(-1, sorted_idx.argsort(-1), sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), 1).item()

    # First generated token
    next_id = sample(logits[:, -1], ids)
    generated = [next_id]
    print(tokenizer.decode([next_id]), end="", flush=True)

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
        print(tokenizer.decode([next_id]), end="", flush=True)
    t_dec_total = time.perf_counter() - t_dec

    n_gen = len(generated)
    print(f"\n\n[prefill: {len(ids)} tok @ {len(ids) / t_prefill:.0f} t/s"
          f" | decode: {n_gen} tok @ {max(1, n_gen - 1) / t_dec_total:.1f} t/s]")
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────

def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nanollama — educational LLM inference")
    parser.add_argument("--prompt", required=True, help="input prompt")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps (auto-detected)")
    parser.add_argument("--temp", type=float, default=0.7, help="sampling temperature (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=50, help="top-k filtering: keep k most likely tokens, 0=disabled (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.9, help="top-p nucleus sampling threshold, 1.0=disabled (default: 0.9)")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="repetition penalty: 1.0=off, 1.1=mild, 1.3=strong (default: 1.1)")
    parser.add_argument("--max-tokens", type=int, default=200, help="max tokens to generate (default: 200)")
    args = parser.parse_args()

    device = args.device or auto_device()
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}\n")

    model, tokenizer = load_model(args.model, device)
    print()
    generate(model, tokenizer, args.prompt, args.max_tokens, args.temp, args.top_k,
             args.top_p, args.repeat_penalty)
