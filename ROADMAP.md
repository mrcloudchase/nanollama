# nanollama Roadmap

Features to add incrementally on top of the ~400-line starting point. Each item
is a self-contained exercise that teaches a new concept. Ordered by complexity —
start from the top and work your way down.

**Current state:** Phase 1 complete. Working on Phase 2.

---

## Phase 0: Core Inference (DONE)

- [x] **Transformer architecture** — RMSNorm, RoPE, GQA, SwiGLU in pure PyTorch
- [x] **KV-Cache** — Efficient autoregressive generation with cached key/value states
- [x] **Temperature sampling** — Greedy (temp=0) and stochastic (temp>0) token selection
  - Output sample (`--prompt "The meaning of life is" --top-k 0 --top-p 1.0 --repeat-penalty 1.0`):
    ```
    to find your purpose, and your purpose is to make a difference in
    the world.

    2. Life is a journey, not a destination: This line comes from the
    movie "Into the Woods" and is a great reminder that life is a
    rollercoaster ride with ups and downs.

    3. The present moment: This line comes from the song "My Heart Will
    Go On" and is a great reminder to live in the present moment...

    [prefill: 6 tok @ 29 t/s | decode: 200 tok @ 8.9 t/s]
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
  - Output sample (`--prompt "The meaning of life is" --top-k 50 --top-p 1.0 --repeat-penalty 1.0`):
    ```
    not just a question of finding the right job, home or partner. It's
    also about living the present moment to its fullest potential.

    2. You are here to learn and grow. The journey is as important as
    the destination.

    3. Life is not about being rich or famous but about embracing the
    present and living the life you were born to live.

    4. The path to happiness is not by seeking it, but by being content
    with where you are, doing what you do, and being grateful for the
    experience.

    5. Life is about living your dreams, not waiting for them to happen.

    6. The greatest wealth is that which comes from within, and the most
    valuable possession is your mind.

    7. Life is a journey, not a destination. The key to success is to
    keep moving forward even when things seem challenging.

    8. The only thing that matters in life is the quality of

    [prefill: 6 tok @ 5 t/s | decode: 200 tok @ 10.0 t/s]
    ```
    Coherent output, but without top-p or repetition penalty the model tends
    to repeat patterns (numbered lists, similar phrasing).

- [x] **Top-p (nucleus) sampling** — Keep the smallest set of tokens whose
  cumulative probability exceeds p. Unlike top-k, this adapts to the shape
  of the distribution — fewer tokens when confident, more when uncertain.
  - *What you'll learn*: Cumulative distributions, adaptive filtering,
    why top-p often beats top-k.
  - Output sample (`--prompt "The meaning of life is" --top-k 50 --top-p 0.9 --repeat-penalty 1.0`):
    ```
    to pursue your passion. If you can't find it, make it.

    [prefill: 6 tok @ 23 t/s | decode: 18 tok @ 10.3 t/s]
    ```
    With top-p layered on, the model produces concise output and stops
    naturally at EOS instead of rambling through numbered lists.

- [x] **Repetition penalty** — Reduce the probability of tokens that already
  appeared. Prevents the model from getting stuck in loops.
  - *What you'll learn*: Post-hoc logit manipulation, the repetition problem
    in autoregressive models.
  - Output sample (`--prompt "The meaning of life is" --repeat-penalty 1.1`):
    ```
    to be yourself in every way, while the purpose of existence is to
    live your truth. 5. "I am not a slave, but I have become one" The
    meaning of life is to find oneself and not to be found. These quotes
    from famous people on finding self-identity are inspirational and
    thought-provoking. They remind us that we all have our own unique
    journey towards self-discovery, and that we should celebrate our
    differences.

    [prefill: 6 tok @ 52 t/s | decode: 97 tok @ 9.1 t/s]
    ```
    Combined with top-k and top-p (defaults), repetition penalty produces
    varied, non-repetitive output that terminates naturally.

---

## Phase 2: Chat & Interaction

- [x] **Chat template** — Wrap user input in the model's expected chat format
  (e.g., ChatML's `<|user|>`, `<|assistant|>` tags) so chat-tuned models
  respond properly instead of continuing raw text.
  - *What you'll learn*: Why prompt format matters, how chat fine-tuning works.
  - Output sample (`--prompt "What is the capital of France?" --chat`):
    ```
    The capital of France is Paris.

    [prefill: 23 tok @ 113 t/s | decode: 8 tok @ 9.5 t/s]
    ```
    With the chat template, the model gives a direct answer instead of
    continuing raw text.

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
