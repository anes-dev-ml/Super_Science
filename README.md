# Super_Science

**An experimental C++17 project for learning numerical computing, Transformer
architecture, and language-model inference from the inside out.**

The aim is not to create the most capable AI assistant or compete with mature
inference runtimes. The aim is to understand the low-level calculations and
engineering decisions behind them, implement them in C++, and measure whether
they work.

This is an **educational, source-available engineering project**. It includes
original numerical and inference code, alongside an acknowledged JSON parsing
dependency. It does **not** train a new language model.

## What is implemented

| Area | Implementation |
| --- | --- |
| Numerical core | Row-major `float` tensors, shape/stride indexing, linear algebra, activations |
| Matrix multiplication | Basic checked, direct indexing, `i,k,j`, and tiled implementations |
| Transformer experiments | RMSNorm, causal attention, RoPE, SwiGLU, grouped-query attention |
| Decoding experiments | Full, cached, and buffered KV-cache generation paths |
| Model-specific runtime | Dense Qwen3 CPU float32 token-to-logits inference |
| Tokenization | C++ byte-level BPE and a Qwen3 chat prompt formatter |
| Checkpoint reader | SafeTensors header validation; F32, F16, BF16 converted to float32 |
| Interfaces | Command-line chat, token/logits diagnostic CLI, optional Qt 6 desktop chat |
| Optimization | Optional OpenMP for large matrix-vector operations |
| Verification | Synthetic tests, validation tests, offline Hugging Face comparison utility |

**Important scope:** the educational tensor/Transformer algorithms and the
model-specific `Qwen3Cpu` implementation are separate execution paths.
They use different RoPE conventions; see [architecture.md](docs/architecture.md).

## Build and test

Requires a **C++17 compiler** and **CMake 3.16 or newer**. The core does not
require Python, Qt, CUDA, or hosted services.

From the project root:

```powershell
cmake -S . -B build -DSUPER_SCIENCE_BUILD_DESKTOP=OFF
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

On Windows, these commands work with a correctly configured MSVC or MinGW
toolchain. A Visual Studio multi-configuration generator places executables
in `build/Release/`; a single-configuration MinGW generator normally uses
`build/`. For a single-configuration generator, set
`-DCMAKE_BUILD_TYPE=Release` at CMake configuration time.

CMake registers six test executables. Their existence does not itself
establish independently verified correctness; see
[validation and reproducibility](docs/validation.md).

## Run the native Qwen3 chat CLI

Provide a **compatible dense Qwen3** model directory containing
`config.json`, `tokenizer.json`, and a single `model.safetensors` file.
Model artifacts are not included here.

```powershell
# Multi-configuration Windows build:
.\build\Release\qwen3_chat.exe models/qwen3-0.6b --prompt "Hello!" --max-new-tokens 64
```

The executable path changes to `build/qwen3_chat.exe` for typical
single-configuration MinGW builds.

The CLI performs local CPU inference and supports `--thinking` or `--raw`
prompt formatting. `qwen3_cli` accepts pre-encoded token IDs and can dump
raw logits for numerical comparison against a trusted reference.

**Current limitations:** CPU float32 weights and computation, greedy decoding,
single-file weights, no quantized or GPU inference, no general model-family
compatibility, and no persistent multi-turn KV-cache reuse. Very large models
or long prompts may take substantial RAM and computation time.

## Matrix multiplication experiment

There is a separate **opt-in** benchmark target to compare four approaches,
with result verification before measuring performance:

```powershell
cmake -S . -B build-bench -DSUPER_SCIENCE_BUILD_BENCHMARKS=ON -DSUPER_SCIENCE_BUILD_DESKTOP=OFF
cmake --build build-bench --config Release --parallel 4
.\build-bench\Release\matmul_bench.exe 128 128 128
```

For a single-configuration generator, the executable is normally
`build-bench/matmul_bench.exe`. Report hardware, compiler, build mode,
matrix dimensions, and repeat counts alongside any timing claims. No specific
performance figures are asserted without a measured result.

## Optional Qt 6 desktop chat

The Qt 6 Widgets app is deliberately simple: one in-memory conversation,
streamed text, Stop, and New chat. It uses the **same C++ CPU inference**
implementation, running in a worker thread. Qt is not used by the CLI or
core numerical code.

Install Qt 6 Widgets with the kit matching your compiler, then configure
using that kit's actual path:

```powershell
cmake -S . -B build-desktop -DSUPER_SCIENCE_BUILD_DESKTOP=ON -DCMAKE_PREFIX_PATH="C:/Qt/6.8.3/msvc2022_64"
cmake --build build-desktop --config Release --target super_science_desktop
.\build-desktop\Release\super_science_desktop.exe models/qwen3-0.6b
```

The Qt path is an **example**, not an assumed installation path. Use the
matching Qt command prompt or `windeployqt` if runtime DLLs are not found.
For MinGW, ensure the compiler and Qt kit use compatible toolchains.

The desktop does not save chat history and rebuilds the model's KV cache
on each turn. Long prompt processing can delay Stop and application shutdown.

## Repository map

```text
apps/               # Qwen3 CLIs, optional Qt desktop
include/
  tensor.hpp        # Educational dense tensor core
  matrix.hpp        # Earlier matrix implementation
  inference/        # Transformer, caches, SafeTensors, configuration
  tokenizer/        # Native Qwen3 tokenizer
  qwen3_cpu.hpp     # Model-specific reference runtime
benchmarks/         # Opt-in numerical benchmarks
tests/              # Synthetic and validation test programs
tools/              # Offline Python reference-comparison helpers
docs/               # Architecture and verification instructions
CMakeLists.txt
```

For the mathematics and architecture, read
[docs/architecture.md](docs/architecture.md). For how to test the project
and reproduce measurements, read [docs/validation.md](docs/validation.md).

## Dependencies and licensing

The core inference runtime is C++17; it bundles **nlohmann/json 3.12.0**
under its MIT license. OpenMP is optional. Qt 6 is optional for the desktop.
Two Python scripts can be used for **development-time reference checks only**;
they are not runtime dependencies.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

Copyright © 2026 Anes. All Rights Reserved. The original project code is
**source-available, not open-source licensed**. Viewing its public GitHub
source does not imply permission to reuse or redistribute it. See [LICENSE](LICENSE).

## Status and future exploration

**Experimental.** Before describing the project as independently validated,
record successful native build/test runs, tokenizer golden-token parity,
and full-checkpoint Qwen3 logit comparisons.

Possible later investigations: SIMD kernels, quantization, deeper memory
profiling, and additional numerical reference tests. A production AI assistant,
web tools, and broad model support are intentionally outside this project's scope.
