# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

nanollama is an educational LLM inference engine in a single file (`nanollama.py`, ~750 lines). It implements a LLaMA/Qwen2-architecture transformer from scratch in PyTorch to teach how LLM inference works. It is intentionally minimal — see ROADMAP.md for features to add incrementally.

## Commands

```bash
# Run (venv already exists at .venv/)
source .venv/bin/activate
python nanollama.py --prompt "Hello world" --chat

# Common options
python nanollama.py --prompt "text" --chat --dtype float16 --max-tokens 200
python nanollama.py --interactive --system "You are a helpful assistant."
python nanollama.py --batch-file prompts.txt --chat --dtype float16
python nanollama.py --prompt "text" --chat --quantize q8
python nanollama.py --prompt "text" --chat --compile
```

No tests, no linter, no build step. It's a single-file script.

## Architecture

Everything is in `nanollama.py`, organized top-to-bottom as a dependency chain:

### Model components

- **Config** — dataclass with model hyperparameters (hidden size, head counts, layer count, etc.). Loaded from HuggingFace `config.json` via `Config.from_json()`, which filters to only fields we define. Includes `attention_bias` flag for Qwen2 models.
- **RMSNorm** — Root Mean Square normalization. Like LayerNorm but skips mean-centering — just divides by `sqrt(mean(x²) + eps)` and scales by a learned weight. Computes in float32 for stability regardless of input dtype.
- **QuantizedLinear** — Drop-in `nn.Linear` replacement for Q8 (int8) and Q4 (4-bit) quantization. Stores weights as int8 + per-output-channel absmax scale. Q4 packs two 4-bit values per int8 byte using bit shifts. Dequantizes to float on each forward pass. `quantize_model()` recursively replaces all `nn.Linear` layers.
- **RoPE** (`precompute_rope` + `apply_rope`) — Rotary Positional Embeddings. Precomputes cos/sin tables indexed by position. Uses LLaMA's split-half pairing: dimension pairs are (d, d+dim//2) rather than interleaved. `apply_rope` splits Q/K into halves and rotates each pair by position-dependent angles. Only applied to Q and K (not V). Supports batched position_ids tensor `[B, S]` for padded inputs.
- **Attention** — Grouped-Query Attention (GQA) where multiple Q heads share fewer KV heads (e.g., 32 Q heads, 4 KV heads). KV heads are repeated via `unsqueeze+expand` to match Q head count before computing `softmax(QK^T/sqrt(d))V`. Attention scores always computed in float32 for float16 stability on MPS. KV-cache: lazily allocates full-length cache tensors, supports both int offset (single prompt) and batched position_ids (deterministic loop for left-padded inputs).
- **FFN** — SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`. Three linear projections (no bias). The element-wise multiply of gated and ungated paths lets the network learn which features to pass through.
- **Block** — One transformer layer: pre-norm residual pattern. `x + attn(norm(x))` then `h + ffn(norm(h))`. Pre-norm (normalize before sublayer) is more stable than post-norm for deep networks.
- **Transformer** — Full model. Embedding lookup → N Blocks → final RMSNorm → lm_head linear → logits. Builds causal mask (upper triangular of -inf). Supports optional `position_ids` [B, S] for batched RoPE and `pad_mask` [B, total_seq_len] to block attention to padding positions (uses `torch.where` to avoid `0.0 * -inf = NaN`).

### Inference pipeline

- **load_model()** — Downloads from HuggingFace Hub via `snapshot_download`, loads safetensors weights, strips `model.` prefix from weight names. Auto-detects tied embeddings (copies `embed_tokens` to `lm_head` if missing) and QKV bias (Qwen2). Accepts `dtype` parameter (float32/float16/bfloat16) for model conversion. Returns `(model, tokenizer)`.
- **Chat Template** (`apply_chat_template()`) — Renders conversation messages using the model's Jinja2 chat template from `tokenizer_config.json`. Handles `bos_token`, `eos_token`, and `add_generation_prompt`. Falls back to hardcoded ChatML format if no template found.
- **generate()** — Two-phase single-prompt generation loop:
  - *Prefill*: Process entire prompt in one forward pass (start_pos=0), populates KV-cache for all prompt positions.
  - *Decode*: Generate tokens one at a time. Each step feeds only the new token (start_pos increments), KV-cache provides context. Inner `sample()` function applies: repetition penalty → temperature scaling → top-k filtering → top-p nucleus sampling → multinomial/argmax. Streams each token to stdout by decoding the full generated sequence each step for correct tokenizer spacing.
- **generate_batch()** — Batched multi-prompt generation. Left-pads all prompts to same length. Constructs per-token `position_ids` for correct RoPE despite padding. Builds `pad_mask` indexed by cache positions (not input positions). During decode, each sequence tracks its own cache position independently to avoid position gaps.
- **main** — argparse CLI with 14 flags. Auto-detects best device (cuda > mps > cpu). Supports three modes: single prompt, interactive REPL (multi-turn with context truncation), and batch file. Post-load quantization and torch.compile() applied before generation.

## Key design decisions

- **Single file by design.** Don't split into modules unless the user asks to (that's a ROADMAP item). New features get added to `nanollama.py`.
- **Chat template via `--chat` flag.** Uses the model's Jinja2 template from `tokenizer_config.json` to format conversations. Falls back to ChatML `<|user|>\n{prompt}</s>\n<|assistant|>\n` if no template found. Without `--chat`, prompts are passed raw for text completion. Chat mode skips BOS token (the template provides its own framing).
- **Full sampling pipeline.** Repetition penalty → temperature → top-k → top-p → multinomial sampling. All configurable via CLI flags. `logits.clone()` prevents in-place mutation of model output.
- **Weight names:** HuggingFace prefixes with `model.` (e.g., `model.layers.0.self_attn.q_proj.weight`), our Transformer uses `layers.0.self_attn.q_proj.weight`. The loader strips the prefix. `strict=False` handles tied/missing lm_head weights.
- **KV-cache is lazily allocated** on first forward pass per attention layer, sized to `max_position_embeddings`. Call `model.reset()` between independent generations.
- **Float16 stability:** Attention scores are always computed in float32 (`q.float() @ k.float()`) because MPS accumulates float16 matmuls in half-precision (unlike CUDA), which causes overflow. RMSNorm and softmax also compute in float32.
- **Quantization is post-load.** Model loads in full precision, then `quantize_model()` replaces `nn.Linear` layers with `QuantizedLinear`. No speed benefit without custom kernels (dequantize per forward), but real memory savings.
- **Batched pad_mask tracks cache positions**, not input positions. With left-padding, cache position 0 contains the real first token (not padding), so the mask must reflect what's actually in the cache after the KV write.

## When adding features

Check ROADMAP.md first — features are tracked there with checkboxes. After implementing one, mark it `[x]` in the roadmap and update the "Current state" line at the top. Keep the code educational with concise inline comments explaining the "why."

## Git Workflow

```bash
# Commit changes
git add <files>
git commit -m <commit message>

# Push to remote
git push
```

Always commit and push without co-author tags. Keep commit messages concise — focus on the "why" rather than the "what".
