# nanollama Roadmap

Features added incrementally to `nanollama.py` (currently ~750 lines). Each item
is a self-contained exercise that teaches a new concept. Ordered by complexity —
start from the top and work your way down.

**Current state:** Phase 4 complete. Working on Phase 5.

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

- [x] **Interactive mode** — A REPL loop: prompt → generate → prompt → ...
  with `--interactive` flag.
  - *What you'll learn*: Building simple CLI interfaces, user input handling.
  - Output sample (`--interactive`):
    ```
    Interactive mode — type a message, press Enter to generate.
    Press Ctrl+C or type /exit to quit.

    > What is the capital of France?
    The capital city of France is Paris.

    > Who wrote Romeo and Juliet?
    Romeo and Juliet was written by William Shakespeare in 1597.
    ```

- [x] **Jinja2 chat templates** — Load chat templates from the model's
  `tokenizer_config.json` instead of hardcoding. This is how HuggingFace and
  Ollama support many chat formats with one codebase.
  - *What you'll learn*: Jinja2 templating, how different models use different formats.
  - Output sample (`--prompt "What is the capital of France?" --chat --model HuggingFaceTB/SmolLM2-1.7B-Instruct`):
    ```
    The capital of France is Paris.

    [prefill: 37 tok @ 19 t/s | decode: 8 tok @ 0.2 t/s]
    ```
    Same `--chat` flag, different model — the Jinja2 template automatically
    used SmolLM2's `<|im_start|>` format instead of TinyLlama's `<|user|>`.

- [x] **Multi-turn conversation** — Maintain conversation history across turns
  by concatenating past messages into the prompt.
  - *What you'll learn*: Context window management, when to truncate history.
  - Output sample (`--interactive`):
    ```
    > My name is Alice.
    I am not able to have a name, but I can provide you with
    information and resources related to alice's story...

    > What is my name?
    That is a private question that you should keep to yourself...

    [prefill: 95 tok — includes full conversation history]
    ```
    The second prefill (95 tokens) shows the model sees the entire
    conversation, not just the latest message. Oldest turns are
    dropped when the context window fills up.

- [x] **System prompts** — Allow custom system prompts via `--system` flag.
  - *What you'll learn*: How system prompts steer model behavior.
  - Output sample (`--prompt "Hello" --chat --system "You are a pirate."`):
    ```
    Rendered template:
    <|system|>\nYou are a pirate.</s>\n<|user|>\nHello</s>\n<|assistant|>\n

    [prefill: 45 tok — system message adds tokens to the context]
    ```
    Works with both `--chat` and `--interactive`. In interactive mode,
    the system prompt persists across all turns.

---

## Phase 3: Performance

- [x] **Float16 inference** — Run in half-precision to halve memory. Requires
  careful handling of norms and softmax for numerical stability.
  - *What you'll learn*: Floating-point precision, mixed-precision computation,
    why MPS float16 matmuls need float32 accumulation.
  - Output sample (`--prompt "What is the capital of France?" --chat --dtype float16`):
    ```
    Loaded 28 layers, 1.8B params on mps (float16)

    Okay, so I need to figure out what the capital of France is...

    [prefill: 12 tok @ 17 t/s | decode: 50 tok @ 10.0 t/s]
    ```
    100x speedup over float32 (10.0 vs 0.1 t/s decode) with identical output.
    Attention scores are computed in float32 to avoid overflow on MPS where
    float16 matmuls accumulate in half-precision (unlike CUDA).

- [x] **Basic quantization (Q8)** — Store weights as int8 + scale, dequantize
  during matmul. ~4x memory savings with minimal quality loss.
  - *What you'll learn*: Quantization theory, per-channel absmax scaling,
    memory vs. speed tradeoffs (dequantize is slow without custom kernels).
  - Output sample (`--prompt "What is the capital of France?" --chat --quantize q8`):
    ```
    Quantized: 2.5GB (Q8)   [vs ~7GB float32]

    Okay, so I need to figure out what...
    ```
    Correct output, ~4x memory savings. Decode is slow because dequantize
    happens in Python — real speedup needs fused CUDA/MPS kernels.

- [x] **4-bit quantization (Q4)** — More aggressive compression with block-wise
  scaling. ~8x memory savings.
  - *What you'll learn*: 4-bit packing (two values per int8), nibble
    extraction with bit shifts, quality vs. compression tradeoffs.
  - Output sample (`--quantize q4`):
    ```
    Quantized: 1.7GB (Q4)   [vs ~7GB float32]
    ```
    Two 4-bit values packed into each int8 byte. Unpacking uses arithmetic
    right shift (preserves sign) for the high nibble and shift-left-then-right
    for the low nibble.

- [x] **torch.compile()** — Automatic kernel fusion. Can give 2-3x speedup.
  - *What you'll learn*: PyTorch compiler (tracing, graph capture, fusion).
  - Usage: `--compile` flag. First forward pass is slow (compilation), then
    subsequent passes benefit from fused kernels. Works with all dtype options.

- [x] **Batched generation** — Process multiple prompts simultaneously.
  - *What you'll learn*: Left-padding, position IDs for RoPE, padding masks,
    per-sequence KV cache management, throughput vs. latency tradeoffs.
  - Output sample (`--batch-file prompts.txt --chat --dtype float16`):
    ```
    Batch: 3 prompts

    [batch=3 | prefill: 35 tok @ 10 t/s | decode: 90 tok @ 0.8 t/s]

    --- Prompt 1 ---
    Okay, so I need to figure out what the capital of France is...

    --- Prompt 2 ---
    I need to determine the sum of 2 and 2...

    --- Prompt 3 ---
    Okay, so I'm trying to figure out who wrote "Romeo and Juliet."...
    ```
    Key challenges: left-padding aligns generation positions, position_ids
    give correct RoPE positions despite padding, pad_mask tracks valid
    CACHE positions (not input positions), per-sequence position tracking
    during decode to avoid gaps.

---

## Phase 4: API Server

- [x] **FastAPI server** — HTTP API with `/v1/completions` and `/v1/chat/completions`
  endpoints, plus `/v1/models`. Started with `--serve` flag.
  - *What you'll learn*: REST APIs, request schemas, async Python.
  - Usage: `python nanollama.py --serve --dtype float16 --port 8000`
  - Output sample:
    ```
    Starting server on http://0.0.0.0:8000
    OpenAI-compatible API: http://localhost:8000/v1/chat/completions
    ```

- [x] **SSE streaming** — Stream tokens via Server-Sent Events when `stream=true`.
  - *What you'll learn*: SSE protocol, chunked responses, OpenAI streaming format.
  - Each chunk is `data: {json}\n\n` matching OpenAI's format, ending with
    `data: [DONE]\n\n`.
  - Output sample (`curl ... -d '{"messages":[...],"stream":true}'`):
    ```
    data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

    data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

    data: [DONE]
    ```

- [x] **OpenAI-compatible API** — Full `/v1/chat/completions` compatibility so
  existing tools (LangChain, Open WebUI, curl) work as a drop-in backend.
  - *What you'll learn*: API design, Pydantic models, compatibility layers.

- [x] **Request queuing** — `asyncio.Lock` serializes model access so concurrent
  requests queue safely (the model's KV-cache is mutable shared state).
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
