# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

nanollama is an educational LLM inference engine in a single file (`nanollama.py`, ~1400 lines). It implements a LLaMA/Qwen2-architecture transformer from scratch in PyTorch to teach how LLM inference works. It is intentionally minimal — see ROADMAP.md for features to add incrementally.

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
python nanollama.py --serve --dtype float16 --port 8000

# Model management
python nanollama.py list
python nanollama.py rm deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# GGUF model
python nanollama.py --model model.gguf --tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --prompt "Hello" --chat

# Modelfile config
python nanollama.py --modelfile my.mf --prompt "Hello" --chat
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
- **GGUF Loading** (`load_gguf()`) — Parses GGUF binary format (v2/v3): reads magic/version header, metadata key-value pairs (architecture-prefixed keys for Config fields), and tensor info array. Dequantizes Q4_0 (18-byte blocks: f16 scale + 16 packed nibbles offset by -8), Q4_1 (20-byte blocks: f16 scale + f16 min + 16 packed nibbles), and Q8_0 (34-byte blocks: f16 scale + 32 int8) using numpy. `_map_gguf_name()` converts GGUF names (`blk.0.attn_q.weight`) to our names (`layers.0.self_attn.q_proj.weight`). Auto-downloads from HuggingFace when given `org/repo/filename.gguf` format. Requires `--tokenizer` for a HuggingFace tokenizer since GGUF doesn't include Python-usable tokenizer data. Infers vocab_size from the embedding tensor shape rather than metadata.
- **Chat Template** (`apply_chat_template()`) — Renders conversation messages using the model's Jinja2 chat template from `tokenizer_config.json`. Handles `bos_token`, `eos_token`, and `add_generation_prompt`. Falls back to hardcoded ChatML format if no template found.
- **generate_streaming()** — Generator variant of `generate()` that yields `{"text", "token_id", "finish_reason"}` dicts instead of printing. Same prefill + decode + sampling logic. Used by the API server for both streaming and non-streaming responses.
- **generate()** — Two-phase single-prompt generation loop:
  - *Prefill*: Process entire prompt in one forward pass (start_pos=0), populates KV-cache for all prompt positions.
  - *Decode*: Generate tokens one at a time. Each step feeds only the new token (start_pos increments), KV-cache provides context. Inner `sample()` function applies: repetition penalty → temperature scaling → top-k filtering → top-p nucleus sampling → multinomial/argmax. Streams each token to stdout by decoding the full generated sequence each step for correct tokenizer spacing.
- **generate_batch()** — Batched multi-prompt generation. Left-pads all prompts to same length. Constructs per-token `position_ids` for correct RoPE despite padding. Builds `pad_mask` indexed by cache positions (not input positions). During decode, each sequence tracks its own cache position independently to avoid position gaps.
- **API Server** (`create_app()`) — FastAPI app factory that returns an OpenAI-compatible HTTP server. Endpoints: `POST /v1/chat/completions` (chat with streaming), `POST /v1/completions` (text completion with streaming), `GET /v1/models` (list loaded model). Request/response schemas use Pydantic models matching OpenAI's format. An `asyncio.Lock` serializes model access since the KV-cache is mutable shared state. Streaming uses SSE (`data: {json}\n\n` chunks, ending with `data: [DONE]\n\n`). Generation runs in a thread pool via `asyncio.to_thread` to avoid blocking the event loop.
- **Model Registry** (`cmd_list()`, `cmd_rm()`) — Uses `huggingface_hub.scan_cache_dir()` to list/remove cached models. No separate JSON database — the HF cache is the source of truth. `cmd_rm()` collects all revision hashes and calls `strategy.execute()` after user confirmation.
- **Modelfile** (`parse_modelfile()`) — Ollama-style config DSL parser. `Modelfile` dataclass holds model ID, tokenizer ID, system prompt, and parameters dict. Four directives: `FROM` (model), `TOKENIZER` (HuggingFace tokenizer for GGUF), `PARAMETER key value` (sampling), `SYSTEM text` (system prompt). `--modelfile` flag overrides model, tokenizer, system, and sampling defaults.
- **main** — argparse CLI with 18 flags plus `list`/`rm` subcommands. Subcommands are dispatched before argparse via `sys.argv[1]` check. Auto-detects best device (cuda > mps > cpu). Supports GGUF loading (`.gguf` extension triggers `load_gguf()`), Modelfile config (overrides args), and three generation modes: single prompt, interactive REPL, and batch file. Post-load quantization and torch.compile() applied before generation.

## Key design decisions

- **Single file by design.** Don't split into modules unless the user asks to (that's a ROADMAP item). New features get added to `nanollama.py`.
- **Chat template via `--chat` flag.** Uses the model's Jinja2 template from `tokenizer_config.json` to format conversations. Falls back to ChatML `<|user|>\n{prompt}</s>\n<|assistant|>\n` if no template found. Without `--chat`, prompts are passed raw for text completion. Chat mode skips BOS token (the template provides its own framing).
- **Full sampling pipeline.** Repetition penalty → temperature → top-k → top-p → multinomial sampling. All configurable via CLI flags. `logits.clone()` prevents in-place mutation of model output.
- **Weight names:** HuggingFace prefixes with `model.` (e.g., `model.layers.0.self_attn.q_proj.weight`), our Transformer uses `layers.0.self_attn.q_proj.weight`. The loader strips the prefix. `strict=False` handles tied/missing lm_head weights.
- **KV-cache is lazily allocated** on first forward pass per attention layer, sized to `max_position_embeddings`. Call `model.reset()` between independent generations.
- **Float16 stability:** Attention scores are always computed in float32 (`q.float() @ k.float()`) because MPS accumulates float16 matmuls in half-precision (unlike CUDA), which causes overflow. RMSNorm and softmax also compute in float32.
- **Quantization is post-load.** Model loads in full precision, then `quantize_model()` replaces `nn.Linear` layers with `QuantizedLinear`. No speed benefit without custom kernels (dequantize per forward), but real memory savings.
- **Batched pad_mask tracks cache positions**, not input positions. With left-padding, cache position 0 contains the real first token (not padding), so the mask must reflect what's actually in the cache after the KV write.
- **GGUF dequantizes to float at load time**, not per-forward-pass. This uses more memory than keeping weights quantized, but keeps the model code simple — the Transformer class doesn't need to know about GGUF types. Post-load `--quantize` can re-compress if needed.
- **GGUF tokenizer comes from HuggingFace** via `--tokenizer`. Implementing a full BPE tokenizer from GGUF metadata is complex and out of scope for an educational project.
- **Model registry uses the HF cache directly** via `scan_cache_dir()` — no separate JSON database to keep in sync. The HF cache is always the source of truth.
- **Subcommands (`list`/`rm`) dispatch before argparse** by checking `sys.argv[1]`. This avoids conflicting with `--flag` syntax since subcommands don't start with `--`.

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
