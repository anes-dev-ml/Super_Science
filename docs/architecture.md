# Architecture

Super_Science is an experimental **C++17 CPU inference and numerical-computing**
project. Its purpose is to expose the mathematics, memory layout, and execution
steps behind a Transformer language model. It is **not** an optimized replacement
for established LLM runtimes.

## Two related execution paths

1. **Educational tensor / Transformer path**:
   `include/tensor.hpp` and `include/inference/{transformer,gqa,kv_cache}.hpp`
   implement general learning exercises: dense tensors, full-sequence attention,
   KV-cache variants, grouped-query attention (GQA), and several generation paths.
2. **Model-specific Qwen3 CPU path**:
   `include/qwen3_cpu.hpp` loads a supported Qwen3 dense checkpoint and performs
   token-by-token float32 inference. This is the path used by the chat CLI and
   optional desktop UI.

These paths are related but **not interchangeable**. In particular, the
educational `Tensor::applyRoPE` uses a conventional adjacent-pair rotation
with a fixed base of 10,000. The Qwen3 CPU path uses half-vector pairing and
the checkpoint's `rope_theta` setting, matching the intended Qwen3 layout.

## Tensor representation

`Tensor` owns a `std::vector<float>` in contiguous **row-major** order, plus
shape and stride vectors. For a 2-D tensor with width `C`, element `(r, c)`
occupies linear offset `r * C + c`. General N-D indexing uses

```text
offset = sum(coordinate[i] * stride[i])
```

Checked `at(...)` accesses validate rank and bounds. `dataPtr()` exposes
contiguous storage for performance-sensitive kernels. Raw pointer access is
only safe after validating shapes and computed indices.

This is a deliberately small tensor type: float32 values, positive dimensions,
no broadcasting, no automatic differentiation, no GPU device abstraction.

### Matrix-multiplication experiments

The `Tensor` API includes four explicit strategies, retained for comparison:

- `matmul`: simple `i,j,k` loops with checked multidimensional indexing
- `matmulDirect`: same loop order, direct contiguous offsets
- `matmulIKJ`: `i,k,j` loop order for sequential writes to output rows
- `matmulBlocked`: tiled `i,k,j` traversal with a 32-element block size

They express the same matrix product but differ in indexing overhead and
memory locality. Performance depends on matrix dimensions, CPU, compiler flags,
cache, and thread scheduling. Benchmark claims must report that environment.

## Transformer math

Given query, key, and value matrices, scaled dot-product attention is

```text
scores = Q * transpose(K) / sqrt(head_dim)
weights = softmax(scores + causal_mask)
output = weights * V
```

The causal mask prevents position `i` from observing future positions
`j > i`. The code also demonstrates:

- **RMSNorm**: `x / sqrt(mean(x^2) + epsilon) * gamma`
- **RoPE**: position-dependent rotations of query and key channels
- **GQA**: multiple query heads sharing a smaller number of key/value heads
- **SwiGLU**: `down(silu(gate(x)) * up(x))`
- **Residual connections** around attention and feedforward blocks

The model-specific `Qwen3Cpu::step` follows:

```text
token ID -> embedding row
  -> [RMSNorm -> Q/K/V -> QK RMSNorm -> RoPE
      -> append K/V cache -> causal GQA -> output projection -> residual
      -> RMSNorm -> SwiGLU feedforward -> residual] x num_layers
  -> final RMSNorm -> tied embedding projection -> logits
```

It uses one token at a time and treats all previously processed tokens as
causally visible. When only building the prompt cache, it avoids calculating
the expensive vocabulary logits except after the final prompt token.

### KV caching

For every layer, the Qwen3-specific engine maintains flattened key and
value vectors. Processing a new token appends its K/V and computes attention
scores against prior positions. This avoids recomputing old Transformer
hidden-state projections during decoding.

However, the desktop application intentionally **restarts the model KV state**
for each new conversation turn and reprocesses the complete chat transcript.
A long conversation therefore increases prompt-prefill cost. This implementation
also uses greedy decoding (argmax), not sampling.

The educational path separately compares uncached, cached, and preallocated
buffered approaches. It is designed for algorithm exploration, not as the
production Qwen3 cache implementation.

## Checkpoint and tokenizer pipeline

`ModelConfig` reads `config.json` and validates key model dimensions.
`readSafeTensorHeader` validates metadata and byte ranges from a single
`model.safetensors` file. `loadSafeTensor` reads F32, F16 or BF16 values and
converts them to float32 `Tensor` instances.

**Format restrictions:** scalar and zero-sized tensors are unsupported by this
Tensor class, even though the SafeTensors format permits them. The Qwen3 loader
expects a particular dense, tied-embedding architecture and a single checkpoint
file; sharded checkpoint indexes and arbitrary model variants are not supported.

`Qwen3Tokenizer` implements byte-level BPE and the project's Qwen3 chat
formatting in C++. `tokenizer.json` is read locally. Exact compatibility with
the upstream tokenizer should be confirmed using golden token-ID vectors, not
only encode/decode round trips.

## Build targets

- `super_science`: header-only interface target
- `qwen3_chat`: text-in/text-out command-line inference
- `qwen3_cli`: token-ID inference and raw-logits diagnostics
- `*_tests`: small deterministic validation programs
- `super_science_desktop`: optional Qt 6 Widgets chat demo
- `matmul_bench`: optional benchmark target, when enabled

OpenMP is an optional optimization for large matrix-vector products. The core
does not call Python or a hosted inference API. Python utilities in `tools/`
are used only for independent offline comparisons.

## Limits and research directions

This project is a **reference implementation**, not a broadly compatible or
production-ready inference runtime. It does not have quantized kernels, GPU
inference, streaming prompt prefill, persistent multi-turn KV reuse, or broad
model-family support. Future exploration could include SIMD, quantization,
kernel profiling, and additional independent numerical tests without expanding
the project into a general AI assistant.

See [validation.md](validation.md) for reproducible checks and the status of
independent model verification.
