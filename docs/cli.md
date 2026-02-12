# CLI Reference

```
python nanollama.py --prompt PROMPT [--model MODEL] [--device DEVICE]
                    [--temp TEMP] [--top-k TOP_K] [--max-tokens MAX_TOKENS]
```

## Required

| Flag | Description |
|------|-------------|
| `--prompt PROMPT` | The input text for the model to continue |

## Optional

| Flag | Default | Description |
|------|---------|-------------|
| `--model MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID or local path to a downloaded model directory |
| `--device DEVICE` | auto-detected | Compute device: `cpu`, `cuda` (NVIDIA GPU), or `mps` (Apple Silicon). Auto-detection priority: cuda > mps > cpu |
| `--temp TEMP` | `0.7` | Sampling temperature. Controls randomness: `0` = greedy (always pick the most likely token), `0.7` = balanced, `1.5` = creative. Higher values flatten the probability distribution |
| `--top-k TOP_K` | `50` | Top-k filtering. Keep only the k most likely tokens before sampling. Set to `0` to disable. Lower values = more focused output |
| `--max-tokens N` | `200` | Maximum number of tokens to generate. Generation stops early if the model produces an end-of-sequence token |

## Examples

Basic usage:

```bash
python nanollama.py --prompt "The meaning of life is"
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

Short output on CPU:

```bash
python nanollama.py --prompt "Hello" --max-tokens 30 --device cpu
```

Using a different model:

```bash
python nanollama.py --prompt "Hello" --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

## Output format

nanollama streams tokens to stdout as they're generated, then prints performance stats:

```
[prefill: 6 tok @ 28 t/s | decode: 29 tok @ 10.6 t/s]
```

- **prefill** — Processing the prompt (tokens per second). Faster because the whole prompt is batched.
- **decode** — Generating new tokens (tokens per second). Slower because tokens are generated one at a time.
