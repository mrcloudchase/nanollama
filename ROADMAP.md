# nanollama Roadmap

Features to add incrementally on top of the ~300-line starting point. Each item
is a self-contained exercise that teaches a new concept. Ordered by complexity —
start from the top and work your way down.

**Current state:** Phase 0 complete. Working on Phase 1.

---

## Phase 0: Core Inference (DONE)

- [x] **Transformer architecture** — RMSNorm, RoPE, GQA, SwiGLU in pure PyTorch
- [x] **KV-Cache** — Efficient autoregressive generation with cached key/value states
- [x] **Temperature sampling** — Greedy (temp=0) and stochastic (temp>0) token selection
  - Output sample (`--prompt "The meaning of life is"`):
    ```
    tobea.Arewhatis?Whatwouldlifeandtheistheoflifeandthemeaningto.
    Thisisthemeaning2020
    Iam
    Livesisthemeaning,butours,andyouhaveallthemeaningarewe,orthethe1andlife,andIlikethetoandthewilltheinthe
    theHandwhatsandthat
    how."isaTheH,isyourofthetheAndisnothing,andallthehis,lovethe.
    the.thathishavetoyouwhatofthathis.Bming.Andnowas.Inbeingin.beingthea,allthehave,areis,thinkso.,ess,ofin.Heseinessforthe.peopleareareintheIwereaesetheeseare.and.of,theyingthe.I.andinicalthe:thethearetheisetheisingaandthe

    [prefill: 6 tok @ 28 t/s | decode: 200 tok @ 10.1 t/s]
    ```
- [x] **Weight loading** — Download and load HuggingFace safetensors with weight name mapping
- [x] **Tokenizer** — Wrapper around HuggingFace AutoTokenizer
- [x] **Streaming output** — Print tokens as they're generated with perf stats
- [x] **CLI args** — `--prompt`, `--model`, `--device`, `--temp`, `--max-tokens` via argparse

---

## Phase 1: Better Sampling

- [x] **Top-k filtering** — Keep only the k most likely tokens, discard the rest.
  Simplest way to remove garbage from the tail of the distribution.
  - *What you'll learn*: How filtering shapes token distributions, `torch.topk`.
  - Output sample (`--prompt "The meaning of life is" --top-k 50`):
    ```
    tolivelifeasitislife.Itmakesiteasierforyoutoliveonthisistobe4meetheaven.</s>

    [prefill: 6 tok @ 28 t/s | decode: 29 tok @ 10.6 t/s]
    ```

- [ ] **Top-p (nucleus) sampling** — Keep the smallest set of tokens whose
  cumulative probability exceeds p. Unlike top-k, this adapts to the shape
  of the distribution — fewer tokens when confident, more when uncertain.
  - *What you'll learn*: Cumulative distributions, adaptive filtering,
    why top-p often beats top-k.

- [ ] **Repetition penalty** — Reduce the probability of tokens that already
  appeared. Prevents the model from getting stuck in loops.
  - *What you'll learn*: Post-hoc logit manipulation, the repetition problem
    in autoregressive models.

---

## Phase 2: Chat & Interaction

- [ ] **Chat template** — Wrap user input in the model's expected chat format
  (e.g., ChatML's `<|user|>`, `<|assistant|>` tags) so chat-tuned models
  respond properly instead of continuing raw text.
  - *What you'll learn*: Why prompt format matters, how chat fine-tuning works.

- [ ] **Interactive mode** — A REPL loop: prompt → generate → prompt → ...
  with `--interactive` flag.
  - *What you'll learn*: Building simple CLI interfaces, user input handling.

- [ ] **Jinja2 chat templates** — Load chat templates from the model's
  `tokenizer_config.json` instead of hardcoding. This is how HuggingFace and
  Ollama support many chat formats with one codebase.
  - *What you'll learn*: Jinja2 templating, how different models use different formats.

- [ ] **Multi-turn conversation** — Maintain conversation history across turns
  by concatenating past messages into the prompt.
  - *What you'll learn*: Context window management, when to truncate history.

- [ ] **System prompts** — Allow custom system prompts via `--system` flag.
  - *What you'll learn*: How system prompts steer model behavior.

---

## Phase 3: Performance

- [ ] **Float16 inference** — Run in half-precision to halve memory. Requires
  careful handling of norms and softmax for numerical stability.
  - *What you'll learn*: Floating-point precision, mixed-precision computation.

- [ ] **Basic quantization (Q8)** — Store weights as int8 + scale, dequantize
  during matmul. ~4x memory savings with minimal quality loss.
  - *What you'll learn*: Quantization theory, scale/zero-point, memory tradeoffs.

- [ ] **4-bit quantization (Q4)** — More aggressive compression with block-wise
  scaling. ~8x memory savings.
  - *What you'll learn*: Block quantization, GPTQ vs RTN vs AWQ approaches.

- [ ] **torch.compile()** — Automatic kernel fusion. Can give 2-3x speedup.
  - *What you'll learn*: PyTorch compiler (tracing, graph capture, fusion).

- [ ] **Batched generation** — Process multiple prompts simultaneously.
  - *What you'll learn*: Padding, masking, throughput vs. latency tradeoffs.

---

## Phase 4: API Server

- [ ] **FastAPI server** — HTTP API with a `/v1/completions` endpoint.
  - *What you'll learn*: REST APIs, request schemas, async Python.

- [ ] **SSE streaming** — Stream tokens via Server-Sent Events.
  - *What you'll learn*: SSE protocol, chunked responses, OpenAI streaming format.

- [ ] **OpenAI-compatible API** — Full `/v1/chat/completions` compatibility so
  existing tools (LangChain, etc.) work out of the box.
  - *What you'll learn*: API design, compatibility layers.

- [ ] **Request queuing** — Handle concurrent requests with a queue/worker pattern.
  - *What you'll learn*: Concurrency, asyncio, resource management.

---

## Phase 5: Model Management

- [ ] **Model registry** — Local JSON database of downloaded models. Support
  `list` and `rm` subcommands.
  - *What you'll learn*: CLI subcommands, simple database design.

- [ ] **Modelfile** — Config format (like Ollama's) to define model + system
  prompt + sampling parameters.
  - *What you'll learn*: Configuration DSL parsing.

- [ ] **GGUF loading** — Load GGUF format models (used by llama.cpp / Ollama).
  - *What you'll learn*: Binary format parsing, GGUF structure.

---

## Phase 6: Advanced

- [ ] **Multi-file refactor** — Split `nanollama.py` into a proper Python
  package (`model.py`, `sampler.py`, `generate.py`, etc.) once the codebase
  outgrows a single file.
  - *What you'll learn*: Python packaging, module organization.

- [ ] **Speculative decoding** — Small draft model proposes tokens, large model
  verifies in one pass. 2-3x speedup with no quality loss.
  - *What you'll learn*: Draft-verify paradigm, acceptance sampling.

- [ ] **Structured output (JSON mode)** — Constrain generation to valid JSON
  using grammar-guided logit masking.
  - *What you'll learn*: Constrained decoding, finite state machines.

- [ ] **Tool calling** — Model outputs structured tool invocations that get
  executed and fed back.
  - *What you'll learn*: Agent patterns, structured output parsing.

- [ ] **Embedding extraction** — Expose internal representations as embeddings
  for retrieval/similarity.
  - *What you'll learn*: Pooling strategies, cosine similarity, RAG basics.

- [ ] **LoRA adapter loading** — Load LoRA fine-tuning adapters on top of a
  base model.
  - *What you'll learn*: Low-rank decomposition, parameter-efficient fine-tuning.
