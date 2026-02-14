<p align="center">
  <img src="media/logo.png" alt="nanollama" width="200">
</p>

---

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/mrcloudchase/nanollama)](https://github.com/mrcloudchase/nanollama/stargazers)

An educational LLM inference engine in ~750 lines of PyTorch.

nanollama is a single-file, from-scratch implementation of LLM inference. It loads a real model from HuggingFace, runs the full transformer forward pass, and generates text — all in code you can read top to bottom in one sitting.

## What you'll learn

By reading `nanollama.py`, you'll understand:

- **RMSNorm** — How LLMs normalize activations (and why it differs from LayerNorm)
- **Rotary Positional Embeddings (RoPE)** — How position is encoded by rotating dimension pairs
- **Grouped-Query Attention (GQA)** — How multiple query heads share key/value heads to save memory
- **KV-Cache** — Why caching past keys/values makes generation fast (prefill vs. decode)
- **SwiGLU FFN** — How gated feed-forward networks work
- **Weight loading** — How pre-trained weights map onto a model architecture
- **Sampling** — How temperature, top-k, top-p, and repetition penalty control token selection
- **Chat templates** — How Jinja2 templates format conversations for different models
- **Quantization** — How Q8/Q4 integer quantization saves memory (per-channel absmax, 4-bit packing)
- **Batched inference** — How left-padding, position IDs, and padding masks enable multi-prompt generation

## Quick start

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python nanollama.py --prompt "What is the capital of France?" --chat
```

The first run downloads the model from HuggingFace (~3GB). Subsequent runs use the cache.
Default model: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B). Use `--model` to try others.

See [docs/cli.md](docs/cli.md) for the full CLI reference, all flags, and examples.

## Reading the code

`nanollama.py` is organized top-to-bottom — each section builds on the one above it:

| Section | Lines | What it teaches |
|---------|-------|-----------------|
| `Config` | ~20 | Model hyperparameters (layers, heads, dimensions) |
| `RMSNorm` | ~13 | Normalization without mean-centering |
| `QuantizedLinear` | ~60 | Q8/Q4 weight quantization with per-channel scaling |
| `RoPE` | ~35 | Positional encoding via rotation (supports batched position IDs) |
| `Attention` | ~65 | GQA + KV-cache with batched position support |
| `FFN` | ~13 | SwiGLU gated feed-forward |
| `Block` | ~15 | Pre-norm residual: norm → attn → + → norm → ffn → + |
| `Transformer` | ~60 | Embed → N blocks → norm → logits (with padding mask support) |
| `load_model` | ~35 | Download + load safetensors weights, dtype conversion |
| `Chat Template` | ~30 | Jinja2 template rendering for any model's chat format |
| `generate` | ~80 | Prefill/decode loop with streaming and sampling |
| `generate_batch` | ~135 | Batched generation with left-padding and per-sequence tracking |
| `main` | ~120 | Device detection, argparse CLI, interactive/batch modes |

## How it compares to Ollama

| Feature | Ollama | nanollama |
|---------|--------|-----------|
| Language | Go + C (llama.cpp) | Pure Python + PyTorch |
| Speed | Highly optimized | Educational, not optimized |
| Quantization | GGUF Q4/Q8/etc. | Float16/Q8/Q4 |
| Model format | GGUF | HuggingFace safetensors |
| API server | Yes (OpenAI-compatible) | No |
| Code size | ~100K lines | ~750 lines |
| Goal | Production inference | Learning |

## What's next

See [ROADMAP.md](ROADMAP.md) for features to add incrementally. Each item is a self-contained exercise that teaches a new concept — start from the top and work down.
