# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

nanollama is an educational LLM inference engine in a single file (`nanollama.py`, ~300 lines). It implements a LLaMA-architecture transformer from scratch in PyTorch to teach how LLM inference works. It is intentionally minimal — see ROADMAP.md for features to add incrementally.

## Commands

```bash
# Run (venv already exists at .venv/)
source .venv/bin/activate
python nanollama.py --prompt "Hello world"

# Options
python nanollama.py --prompt "text" --chat --model MODEL_ID --device cpu --temp 0.7 --max-tokens 200
```

No tests, no linter, no build step. It's a single-file script.

## Architecture

Everything is in `nanollama.py`, organized top-to-bottom as a dependency chain:

### Model components

- **Config** — dataclass with model hyperparameters (hidden size, head counts, layer count, etc.). Loaded from HuggingFace `config.json` via `Config.from_json()`, which filters to only fields we define.
- **RMSNorm** — Root Mean Square normalization. Like LayerNorm but skips mean-centering — just divides by `sqrt(mean(x²) + eps)` and scales by a learned weight. Computes in float32 for stability regardless of input dtype.
- **RoPE** (`precompute_rope` + `apply_rope`) — Rotary Positional Embeddings. Precomputes cos/sin tables indexed by position. Uses LLaMA's split-half pairing: dimension pairs are (d, d+dim//2) rather than interleaved. `apply_rope` splits Q/K into halves and rotates each pair by position-dependent angles. Only applied to Q and K (not V). Makes attention scores depend on relative position.
- **Attention** — Grouped-Query Attention (GQA) where multiple Q heads share fewer KV heads (e.g., 32 Q heads, 4 KV heads). KV heads are repeated via `unsqueeze+expand` to match Q head count before computing `softmax(QK^T/sqrt(d))V`. Includes KV-cache: lazily allocates full-length cache tensors, writes new K/V at `pos`, reads all cached K/V up to `pos+seq_len`.
- **FFN** — SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`. Three linear projections (no bias). The element-wise multiply of gated and ungated paths lets the network learn which features to pass through.
- **Block** — One transformer layer: pre-norm residual pattern. `x + attn(norm(x))` then `h + ffn(norm(h))`. Pre-norm (normalize before sublayer) is more stable than post-norm for deep networks.
- **Transformer** — Full model. Embedding lookup → N Blocks → final RMSNorm → lm_head linear → logits. Builds causal mask (upper triangular of -inf) to prevent attending to future tokens. RoPE cos/sin tables are precomputed in `__init__` and moved to device in `forward`.

### Inference pipeline

- **load_model()** — Downloads from HuggingFace Hub via `snapshot_download`, loads safetensors weights, strips `model.` prefix from weight names, returns `(model, tokenizer)`.
- **Chat Template** (`apply_chat_template()`) — Wraps user prompt in ChatML format for chat-tuned models.
- **generate()** — Two-phase generation loop:
  - *Prefill*: Process entire prompt in one forward pass (start_pos=0), populates KV-cache for all prompt positions.
  - *Decode*: Generate tokens one at a time. Each step feeds only the new token (start_pos increments), KV-cache provides context. Inner `sample()` function applies: repetition penalty → temperature scaling → top-k filtering → top-p nucleus sampling → multinomial/argmax. Streams each token to stdout.
- **main** — argparse CLI, auto-detects best device (cuda > mps > cpu).

## Key design decisions

- **Single file by design.** Don't split into modules unless the user asks to (that's a ROADMAP item). New features get added to `nanollama.py`.
- **Chat template via `--chat` flag.** Wraps prompt in ChatML format (`<|user|>\n{prompt}</s>\n<|assistant|>\n`). Without `--chat`, prompts are passed raw for text completion. Chat mode skips BOS token (the template provides its own framing).
- **Full sampling pipeline.** Repetition penalty → temperature → top-k → top-p → multinomial sampling. All configurable via CLI flags.
- **Weight names:** HuggingFace prefixes with `model.` (e.g., `model.layers.0.self_attn.q_proj.weight`), our Transformer uses `layers.0.self_attn.q_proj.weight`. The loader strips the prefix. `strict=False` handles tied/missing lm_head weights.
- **KV-cache is lazily allocated** on first forward pass per attention layer, sized to `max_position_embeddings`. Call `model.reset()` between independent generations.

## When adding features

Check ROADMAP.md first — features are tracked there with checkboxes. After implementing one, mark it `[x]` in the roadmap and update the "Current state" line at the top. Keep the code educational with concise inline comments explaining the "why."
