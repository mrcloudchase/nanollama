# CLI Reference

```
python nanollama.py list
python nanollama.py rm MODEL_ID
python nanollama.py --prompt PROMPT [--chat] [--model MODEL] [--device DEVICE]
                    [--dtype DTYPE] [--quantize Q8|Q4] [--compile]
                    [--temp TEMP] [--top-k TOP_K] [--top-p TOP_P]
                    [--repeat-penalty PENALTY] [--max-tokens MAX_TOKENS]
                    [--modelfile PATH] [--tokenizer MODEL_ID]
python nanollama.py --interactive [--model MODEL] [--device DEVICE] [...]
python nanollama.py --batch-file FILE [--chat] [--model MODEL] [...]
python nanollama.py --serve [--port PORT] [--model MODEL] [--device DEVICE] [...]
```

## Subcommands

| Command | Description |
|---------|-------------|
| `list` | List all cached HuggingFace models with size and last modified time |
| `rm MODEL_ID` | Remove a cached model (prompts for confirmation) |

## Required (one of)

| Flag | Description |
|------|-------------|
| `--prompt PROMPT` | The input text for the model to continue |
| `--interactive` | Interactive REPL mode — type prompts, get responses, repeat. Implies `--chat` |
| `--batch-file FILE` | File with one prompt per line for batched generation |
| `--serve` | Start OpenAI-compatible API server instead of generating |

## Optional

| Flag | Default | Description |
|------|---------|-------------|
| `--model MODEL` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | HuggingFace model ID or local path to a downloaded model directory |
| `--device DEVICE` | auto-detected | Compute device: `cpu`, `cuda` (NVIDIA GPU), or `mps` (Apple Silicon). Auto-detection priority: cuda > mps > cpu |
| `--temp TEMP` | `0.7` | Sampling temperature. Controls randomness: `0` = greedy (always pick the most likely token), `0.7` = balanced, `1.5` = creative. Higher values flatten the probability distribution |
| `--top-k TOP_K` | `50` | Top-k filtering. Keep only the k most likely tokens before sampling. Set to `0` to disable. Lower values = more focused output |
| `--top-p TOP_P` | `0.9` | Top-p (nucleus) sampling. Keep the smallest set of tokens whose cumulative probability exceeds p. Set to `1.0` to disable. Adapts to the distribution — fewer tokens when confident, more when uncertain |
| `--repeat-penalty P` | `1.1` | Repetition penalty. Reduces probability of tokens that already appeared. `1.0` = off, `1.1` = mild, `1.3` = strong |
| `--chat` | off | Wrap the prompt using the model's Jinja2 chat template (loaded from `tokenizer_config.json`) so chat-tuned models respond as a conversation instead of continuing raw text. Falls back to ChatML format if no template found |
| `--system PROMPT` | none | System prompt to steer model behavior. Used with `--chat` or `--interactive`. Persists across all turns in interactive mode |
| `--max-tokens N` | `200` | Maximum number of tokens to generate. Generation stops early if the model produces an end-of-sequence token |
| `--dtype DTYPE` | `float32` | Model precision: `float32`, `float16`, or `bfloat16`. Half-precision halves memory and dramatically improves speed. Attention scores are always computed in float32 for numerical stability |
| `--quantize Q` | none | Post-load weight quantization: `q8` (int8, ~4x memory savings) or `q4` (4-bit packed, ~8x savings). Weights are dequantized to float during each forward pass |
| `--compile` | off | Use `torch.compile()` for automatic kernel fusion. First forward pass is slow (compilation), subsequent passes are faster |
| `--port PORT` | `8000` | Server port (used with `--serve`) |
| `--tokenizer MODEL_ID` | none | HuggingFace tokenizer ID. Required when loading GGUF files (GGUF doesn't include a Python-usable tokenizer) |
| `--modelfile PATH` | none | Path to a Modelfile config. Overrides `--model`, `--system`, and sampling defaults with values from the file |

## Examples

Basic usage:

```bash
python nanollama.py --prompt "The meaning of life is"
```

Chat mode (model responds as a conversation):

```bash
python nanollama.py --prompt "What is the capital of France?" --chat
```

Interactive REPL (type prompts, get responses, repeat):

```bash
python nanollama.py --interactive
```

With a system prompt to steer behavior:

```bash
python nanollama.py --interactive --system "You are a helpful coding assistant."
```

Greedy decoding (deterministic, always picks the highest probability token):

```bash
python nanollama.py --prompt "What is 2+2?" --temp 0
```

Creative output with high temperature:

```bash
python nanollama.py --prompt "Once upon a time" --temp 1.5
```

Restrictive sampling (only top 10 tokens considered):

```bash
python nanollama.py --prompt "Hello" --top-k 10
```

Tight nucleus sampling (only tokens in the top 80% probability mass):

```bash
python nanollama.py --prompt "Hello" --top-p 0.8
```

Strong repetition penalty to avoid loops:

```bash
python nanollama.py --prompt "Hello" --repeat-penalty 1.3
```

Short output on CPU:

```bash
python nanollama.py --prompt "Hello" --max-tokens 30 --device cpu
```

Using a different model:

```bash
python nanollama.py --prompt "Hello" --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Half-precision for fast inference (recommended on GPU/MPS):

```bash
python nanollama.py --prompt "Hello" --chat --dtype float16
```

Quantized to int8 (~4x memory savings):

```bash
python nanollama.py --prompt "Hello" --chat --quantize q8
```

4-bit quantization (~8x memory savings):

```bash
python nanollama.py --prompt "Hello" --chat --quantize q4
```

Compiled model with kernel fusion:

```bash
python nanollama.py --prompt "Hello" --chat --compile
```

Batched generation from a file (one prompt per line):

```bash
python nanollama.py --batch-file prompts.txt --chat --dtype float16
```

Start the API server:

```bash
python nanollama.py --serve --dtype float16
```

Start the API server on a custom port:

```bash
python nanollama.py --serve --dtype float16 --port 3000
```

Test the API with curl:

```bash
# Chat completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Text completions
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The meaning of life is","max_tokens":30}'

# List models
curl http://localhost:8000/v1/models
```

List cached models:

```bash
python nanollama.py list
```

Remove a cached model:

```bash
python nanollama.py rm deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

Load a GGUF model (requires a HuggingFace tokenizer):

```bash
python nanollama.py --model path/to/model.gguf --tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --prompt "Hello" --chat --dtype float16
```

Use a Modelfile config:

```bash
# my.mf:
# FROM deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# PARAMETER temperature 0.7
# PARAMETER top_k 50
# SYSTEM You are a helpful coding assistant.

python nanollama.py --modelfile my.mf --prompt "Hello" --chat --dtype float16
```

## Output format

nanollama streams tokens to stdout as they're generated, then prints performance stats:

```
[prefill: 6 tok @ 28 t/s | decode: 29 tok @ 10.6 t/s]
```

- **prefill** — Processing the prompt (tokens per second). Faster because the whole prompt is batched.
- **decode** — Generating new tokens (tokens per second). Slower because tokens are generated one at a time.
