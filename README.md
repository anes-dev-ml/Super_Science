# Super_Science

**A C++ framework for exploring and implementing artificial intelligence systems from scratch.**

Super_Science is a personal engineering project focused on understanding how modern AI systems work at a low level, from tensor operations to large language model inference.

The goal is to build efficient, modular, and maintainable AI infrastructure while developing a deeper understanding of the mathematics and algorithms behind it.

## Features

Current development areas include:

- Matrix and multidimensional tensor operations
- Transformer architecture components
- Large language model inference
- Tokenization and text generation
- KV-cache management
- Model weight loading using SafeTensors

## Project Structure

```text
Super_Science/
├── apps/          # Executable applications
├── include/       # Core framework headers
├── tests/         # Tests and validation
├── tools/         # Development utilities
├── CMakeLists.txt # Build configuration
└── README.md
```

## Technologies

- **Language:** C++
- **Build system:** CMake
- **AI architecture:** Transformer-based language models
- **Model format:** SafeTensors

## Status

**Under active development.**

The framework is experimental. Model compatibility, inference correctness, and performance are still being developed and validated.

## Roadmap

- [ ] Complete and validate Qwen3 inference
- [ ] Improve tensor operations and performance
- [ ] Expand model architecture support
- [ ] Add automated tests and benchmarks
- [ ] Explore neural network training and automatic differentiation

## License

Copyright © 2026 Anes. All Rights Reserved.

This project is source-available for viewing. Reuse, modification, and redistribution require permission from the copyright holder, except where permitted by applicable law.

## Local desktop chat (optional)

The Qt 6 Widgets desktop application is deliberately small: one conversation,
streamed output, Stop, and New chat. It runs the existing C++ CPU inference engine
on a worker thread. It does not use a Python server or a hosted inference API.
It keeps messages in memory only; closing the app clears the conversation.

Install **Qt 6 Widgets** (including the MSVC kit matching your C++ compiler).
On Windows, using the same Visual Studio generator/architecture as your existing
build, configure a separate directory:

```powershell
cmake -S . -B build-desktop -DSUPER_SCIENCE_BUILD_DESKTOP=ON -DCMAKE_PREFIX_PATH="C:/Qt/6.8.3/msvc2022_64"
cmake --build build-desktop --config Release --target super_science_desktop
.\build-desktop\Release\super_science_desktop.exe models/qwen3-0.6b
```

Replace the example Qt path with your installed Qt kit. If Windows cannot load
the Qt runtime DLLs, launch from the matching Qt command prompt or deploy them
using that kit's `windeployqt`. Model weights and tokenizer are not bundled.

**Current constraints:** Qwen3 CPU float32 only; greedy decoding; max 512
generated tokens per reply; one generation at a time; no saved chat history;
Stop is checked during token generation (long prompt prefill can delay it).
Rebuild of KV state for each turn is intentional for correctness and simplicity.
The existing CLI and tests do not require Qt unless the desktop option is enabled.


## CPU inference performance

The float32 reference engine can optionally parallelize large matrix-vector
operations with OpenMP. It also uses independent accumulators for dot products.
Builds without OpenMP still work; this is not a new model format or dependency
for the desktop UI.

With the MinGW/Qt desktop build, reconfigure and rebuild after pulling:

```powershell
cmake -S . -B build-desktop -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER="C:/Qt/Tools/mingw1310_64/bin/g++.exe" -DCMAKE_PREFIX_PATH="C:/Qt/6.12.0/mingw_64" -DSUPER_SCIENCE_BUILD_DESKTOP=ON -DSUPER_SCIENCE_USE_OPENMP=ON
cmake --build build-desktop --parallel 4
ctest --test-dir build-desktop --output-on-failure
```

The configuration should print `OpenMP CPU inference: enabled` if the
toolchain supports it. To compare one and four worker threads on the same
prompt (in a new PowerShell shell with Qt/MinGW available on PATH):

```powershell
$env:OMP_NUM_THREADS = "1"
.\build-desktop\qwen3_chat.exe models/qwen3-0.6b --prompt "Explain what a KV cache does." --max-new-tokens 64
$env:OMP_NUM_THREADS = "4"
.\build-desktop\qwen3_chat.exe models/qwen3-0.6b --prompt "Explain what a KV cache does." --max-new-tokens 64
```

Repeat each run: CPU thermal behavior and process startup cause variability.
The CLI reports model load time, time to first token (including prompt
prefill), total inference time, and subsequent generated tokens per second.
This optimization changes floating-point reduction order slightly; compare
actual model outputs as well as performance. It does not yet optimize prompt
prefill or preserve KV caches between chat turns.
