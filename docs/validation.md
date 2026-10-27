# Validation

This document describes how to build and test Super_Science and how its Qwen3 results were compared with Hugging Face. Reference comparisons use a local model checkpoint; Python is not part of the inference runtime.

## Native tests

From the repository root:

```sh
cmake -S . -B build-audit -DSUPER_SCIENCE_BUILD_DESKTOP=OFF
cmake --build build-audit --config Release --parallel 4
ctest --test-dir build-audit -C Release --output-on-failure
```

The CMake configuration registers six test executables:

| Test | Coverage |
| --- | --- |
| `inference_tests` | Tensor/Transformer operations, attention and KV caching |
| `safetensors_tests` | F32, F16 and BF16 loading |
| `validation_tests` | Invalid dimensions and malformed inputs |
| `qwen3_cpu_tests` | Synthetic inference, attention context, streaming |
| `qwen3_file_tests` | Synthetic checkpoint loading and error handling |
| `tokenizer_tests` | Tokenization, decoding and conversation formatting |

Assertions are explicitly enabled for some test targets in Release configurations. The test suite is also configured to run through GitHub Actions on Windows and Ubuntu.

A complete local CTest result has not yet been provided for review. Build or CI success should be reported only after checking the associated output.

## Reference model

The following comparisons were performed by the project author on October 26, 2026, using a local Qwen3-0.6B checkpoint, a Windows Release build, and Hugging Face Transformers as the reference implementation. Exact checkpoint revision and compiler version were not recorded.

### Numerical logits

Generate C++ logits for a fixed input:

```powershell
.\build-audit\Release\qwen3_cli.exe models/qwen3-0.6b logits 1 2 3 > cpp_logits.txt
python tools/compare_logits.py models/qwen3-0.6b cpp_logits.txt 1 2 3
```

Install the optional reference dependencies with `python -m pip install torch transformers safetensors`. The script handles UTF-8 and PowerShell 5.1 UTF-16 files.

| Input token IDs | Max. absolute error | Mean absolute error | Same top token |
| --- | ---: | ---: | --- |
| `1 2 3` | 0.000028014 | 0.000004843 | Yes |
| `1 2 3 4 5 6 7 8 9 10` | 0.000053167 | 0.000008950 | Yes |
| `100 100 100 100 100 100` | 0.000068665 | 0.000009012 | Yes |

These are full-vocabulary, final-position logit comparisons. Small differences are expected from floating-point accumulation order. The agreement measured here does not imply identical logits for all inputs.

### Tokenizer parity

Run the fixed plain-text comparison suite:

```powershell
python -m pip install transformers tokenizers
python tools/compare_tokenizer.py models/qwen3-0.6b build-audit/Release/tokenizer_cli.exe
```

**Observed:** 15/15 exact token-ID matches with `AutoTokenizer.encode(..., add_special_tokens=False)`.

Cases include ASCII text, code, whitespace, numbers, accented French, Arabic, CJK scripts and emoji. Chat-template parity is not covered by this result.

### KV-cache generation

Three prompts were decoded using greedy selection in both engines:

- C++: cached token-by-token decoding.
- Hugging Face: full-context recomputation at each step, with `use_cache=False`.

| Input | Generated tokens (both implementations) |
| --- | --- |
| `1 2 3 4` | `3 4 3 4` |
| `100 100 100` | `14582 25 1019 40` |
| `9707 0 2585 646` | `358 1477 279 897` |

**Observed:** 3/3 matching sequences, covering 12 generated tokens. This checks generated outputs, not the intermediate cached tensors directly.

## Optional matrix multiplication benchmark

Configure the separate benchmark target:

```sh
cmake -S . -B build-bench -DSUPER_SCIENCE_BUILD_BENCHMARKS=ON -DSUPER_SCIENCE_BUILD_DESKTOP=OFF
cmake --build build-bench --config Release --parallel 4
```

Run `matmul_bench` from the build output directory, for example:

```powershell
.\build-bench\Release\matmul_bench.exe 128 128 128
```

The benchmark checks numerical agreement before timing the alternative implementations. Results should state the CPU, compiler, configuration, matrix dimensions, and measurements. No performance comparison is published here without reproducible timings.

## Coverage and limitations

The reference comparisons cover one checkpoint and a small collection of token sequences. Additional useful tests include longer contexts, chat-template formatting, cancellation behavior, cross-platform numerical accuracy, and error cases for external inputs.

GitHub Actions is configured for Windows and Linux builds and CTest, but workflow configuration alone is not evidence of a passing run. See the repository's [Actions page](../.github/workflows/ci.yml) for the workflow definition and GitHub for live results.
