# Super_Science

**A C++17 implementation of Transformer inference, built from fundamental numerical operations.**

Super_Science explores how language-model inference works beneath high-level frameworks. It includes a tensor implementation, attention and KV-cache experiments, a native Qwen3 tokenizer, and a CPU runtime that can execute supported Qwen3 checkpoints locally.

The project is designed for studying implementation details, testing numerical correctness, and understanding the engineering trade-offs of inference systems. It is not a general-purpose or production inference framework.

## Features

- **Numerical primitives:** contiguous float32 tensors, multidimensional indexing, matrix multiplication, normalization, and activation functions.
- **Transformer components:** causal attention, rotary position embeddings, grouped-query attention, SwiGLU, and KV caching.
- **Qwen3 inference:** token-by-token CPU inference for supported dense Qwen3 checkpoints, using local SafeTensors weights.
- **Tokenizer:** native byte-level BPE, UTF-8 handling, and Qwen3 conversation formatting.
- **Applications:** command-line chat and diagnostics, with an optional Qt 6 desktop interface.
- **Verification:** C++ unit tests and independent comparisons against Hugging Face.

The general-purpose Transformer experiments and the Qwen3 runtime are distinct implementations. Their differences are described in the [architecture documentation](docs/architecture.md).

## Getting started

### Requirements

- CMake 3.16 or later
- A C++17 compiler (MSVC, GCC, or Clang)
- A compatible local Qwen3 checkpoint to run the model

The core does not require Python, CUDA, Qt, or a network service. Python is used only for optional reference comparisons; Qt 6 is needed only for the desktop application.

### Build

From the repository root:

```sh
cmake -S . -B build -DSUPER_SCIENCE_BUILD_DESKTOP=OFF
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

For single-configuration generators, specify `-DCMAKE_BUILD_TYPE=Release` during configuration. With Visual Studio, executables are typically in `build/Release/`; with single-configuration generators they are usually in `build/`.

### Run inference

Place a supported dense Qwen3 checkpoint in a local directory containing:

```text
models/qwen3-0.6b/
├── config.json
├── tokenizer.json
└── model.safetensors
```

Model files are **not distributed** with this repository.

On Windows with a Visual Studio Release build:

```powershell
.\build\Release\qwen3_chat.exe models/qwen3-0.6b --prompt "Hello!" --max-new-tokens 64
```

On Linux or a single-configuration build, use the corresponding executable under `build/`. The CLI also provides `--thinking` and `--raw` prompt modes.

## Implementation

| Component | Location |
| --- | --- |
| Tensor operations | `include/tensor.hpp` |
| Transformer and attention algorithms | `include/inference/` |
| Qwen3 CPU runtime | `include/qwen3_cpu.hpp` |
| Tokenizer | `include/tokenizer/` |
| Applications | `apps/` |
| Tests | `tests/` |
| Benchmarks and reference tools | `benchmarks/`, `tools/` |

See [Architecture](docs/architecture.md) for the numerical and inference design, and [Validation](docs/validation.md) for test commands and comparison methodology.

## Validation

The C++ runtime has been compared against Hugging Face Qwen3-0.6B on a local Windows Release build:

| Check | Observed result |
| --- | --- |
| Final-token logits | Top-token agreement in 3/3 sequences; maximum absolute error below `7e-5` |
| Plain-text tokenization | Identical token IDs in 15/15 test cases |
| Greedy generation with KV caching | Identical generated IDs in 3/3 sequences (12 tokens total) |

These measurements cover the specified test cases, not all possible inputs or model configurations. The [validation report](docs/validation.md) contains exact inputs, errors, reproduction steps, and remaining coverage gaps.

## Scope

Inference currently uses CPU float32 computation and greedy decoding. The supported loader expects a single-file, dense Qwen3 checkpoint; sharded weights, quantized inference, GPU execution, and arbitrary model families are not supported. The desktop application rebuilds the KV cache for each conversation turn.

## License

Copyright © 2026 Anes. All rights reserved.

Super_Science is **source-available**, not open-source licensed. The original code may be viewed, but reuse and redistribution require permission as described in [LICENSE](LICENSE). Third-party components retain their respective licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
