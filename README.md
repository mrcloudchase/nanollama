<p align="center">
  <img src="media/logo.png" alt="nanollama" width="200">
</p>

---

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/mrcloudchase/nanollama)](https://github.com/mrcloudchase/nanollama/stargazers)

An educational LLM inference engine in ~300 lines of PyTorch.

nanollama is a single-file, from-scratch implementation of LLM inference. It loads a real model from HuggingFace, runs the full transformer forward pass, and generates text — all in code you can read top to bottom in one sitting.

## What you'll learn

By reading `nanollama.py`, you'll understand:

- **RMSNorm** — How LLMs normalize activations (and why it differs from LayerNorm)
- **Rotary Positional Embeddings (RoPE)** — How position is encoded by rotating dimension pairs
- **Grouped-Query Attention (GQA)** — How multiple query heads share key/value heads to save memory
- **KV-Cache** — Why caching past keys/values makes generation fast (prefill vs. decode)
- **SwiGLU FFN** — How gated feed-forward networks work
- **Weight loading** — How pre-trained weights map onto a model architecture
- **Sampling** — How temperature controls randomness in token selection

## Quick start

```bash
pip install -r requirements.txt
python nanollama.py --prompt "The meaning of life is"
```

With options:

```bash
python nanollama.py --prompt "What is 2+2?" --temp 0              # greedy (deterministic)
python nanollama.py --prompt "Once upon a time" --temp 1.5        # creative
python nanollama.py --prompt "Hello" --max-tokens 50              # shorter output
python nanollama.py --prompt "Hello" --device cpu                 # force CPU
python nanollama.py --prompt "Hello" --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  # base model
```

The first run downloads the model from HuggingFace. Subsequent runs use the cache.

## Reading the code

`nanollama.py` is organized top-to-bottom — each section builds on the one above it:

| Section | Lines | What it teaches |
|---------|-------|-----------------|
| `Config` | ~20 | Model hyperparameters (layers, heads, dimensions) |
| `RMSNorm` | ~13 | Normalization without mean-centering |
| `RoPE` | ~25 | Positional encoding via rotation |
| `Attention` | ~50 | GQA + KV-cache in one class |
| `FFN` | ~13 | SwiGLU gated feed-forward |
| `Block` | ~15 | Pre-norm residual: norm → attn → + → norm → ffn → + |
| `Transformer` | ~38 | Embed → N blocks → norm → logits |
| `load_model` | ~25 | Download + load safetensors weights |
| `generate` | ~42 | Prefill/decode loop with streaming |
| `main` | ~25 | Device detection, argparse CLI |

## How it compares to Ollama

| Feature | Ollama | nanollama |
|---------|--------|-----------|
| Language | Go + C (llama.cpp) | Pure Python + PyTorch |
| Speed | Highly optimized | Educational, not optimized |
| Quantization | GGUF Q4/Q8/etc. | Float32 only |
| Model format | GGUF | HuggingFace safetensors |
| API server | Yes (OpenAI-compatible) | No |
| Code size | ~100K lines | ~300 lines |
| Goal | Production inference | Learning |

## What's next

See [ROADMAP.md](ROADMAP.md) for features to add incrementally. Each item is a self-contained exercise that teaches a new concept — start from the top and work down.
