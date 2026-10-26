# Verification and reproducibility

This document distinguishes **tests that exist in the repository** from
**results actually reproduced on a machine**. No test or benchmark result is
claimed without a run. Do not infer independent Qwen3 correctness from a
plausible generated sentence.

## 1. Build and run the unit tests

From the repository root:

```powershell
cmake -S . -B build-audit -DSUPER_SCIENCE_BUILD_DESKTOP=OFF
cmake --build build-audit --config Release --parallel 4
ctest --test-dir build-audit -C Release --output-on-failure
```

The project registers six test executables:

| CTest name | What it tests |
| --- | --- |
| `inference_tests` | Educational Transformer, cached/buffered attention, GQA, numerical comparisons |
| `safetensors_tests` | F32/F16/BF16 loading and special float values |
| `validation_tests` | Invalid shapes, cache configurations, and storage ranges |
| `qwen3_cpu_tests` | Synthetic Qwen3 generation, streaming, cancellation, nonzero attention context |
| `qwen3_file_tests` | Synthetic checkpoint loading and missing-weight rejection |
| `tokenizer_tests` | UTF-8/BPE round trips, special tokens, conversation format |

Some tests use `assert()`. The CMake configuration deliberately undefines
`NDEBUG` for these test targets in Release mode. An external CI run and a
local Windows build should confirm those compiler settings.

**Verified test results for the current release: not yet recorded.**

## 2. Matrix multiplication benchmark

Enable the optional target:

```powershell
cmake -S . -B build-bench -DSUPER_SCIENCE_BUILD_BENCHMARKS=ON -DSUPER_SCIENCE_BUILD_DESKTOP=OFF
cmake --build build-bench --config Release --parallel 4
```

Run one of the following, depending on your generator:

```powershell
# Visual Studio / multi-configuration generators
.\build-bench\Release\matmul_bench.exe 128 128 128

# Single-configuration MinGW generator
.\build-bench\matmul_bench.exe 128 128 128
```

The benchmark first compares the four algorithms against a checked-indexing
reference using an absolute-plus-relative tolerance. It then performs one
warmup and five timed runs per algorithm and reports the median.

Dimensions are limited to 1–512 in this educational tool, to avoid accidental
very large checked-indexing benchmarks.

For a meaningful report, record:

- CPU model and operating system
- Compiler name/version, generator, optimization flags and Release mode
- Matrix dimensions, each algorithm's median, and whether correctness passed
- Warmup count, number of repetitions, and other machine load

**Verified timing results for the current release: not yet recorded.**
No speedup ratio should be quoted without the corresponding measurements.

## 3. Compare Qwen3 logits against Hugging Face

This comparison is **optional**, uses Python for verification only, and requires
a compatible locally available dense Qwen3 checkpoint in a single
`model.safetensors` file. It does not change or replace the C++ runtime.

Build `qwen3_cli`, then use a fixed token sequence (example `1 2 3`):

```powershell
.\build-audit\Release\qwen3_cli.exe models/qwen3-0.6b logits 1 2 3 > cpp_logits.txt
python -m pip install torch transformers safetensors
python tools/compare_logits.py models/qwen3-0.6b cpp_logits.txt 1 2 3
```

If the build generator is single-configuration, run
`build-audit/qwen3_cli.exe` instead. Ensure the same token IDs, checkpoint,
and float32 arithmetic configuration are used in both paths.

The comparison script prints:

- maximum and mean absolute error across vocabulary logits
- the top token from each implementation
- top-token agreement

Run on several fixed sequences, including repeated tokens and longer context.
Float32 reduction order may differ across implementations; numerical tolerance
must be justified by measured error, not assumed exact equality.

**Independent full-checkpoint comparison results: not yet recorded.**

## 4. Tokenizer verification

The current C++ tokenizer tests establish round-trip decoding and formatting
on a synthetic vocabulary, but they do **not** verify that its token IDs match
the official Qwen3 tokenizer on realistic inputs.

Before making an exact-tokenizer-compatibility claim, compare fixed golden
token-ID vectors produced by the upstream tokenizer for:

- basic ASCII, punctuation and contractions
- mixed Unicode, emoji, Arabic and accented text
- spaces, line endings and numeric runs
- chat templates with thinking enabled/disabled

**Golden token-ID parity results: not yet recorded.**

## 5. Limitations that affect interpretation

The inference engine runs CPU float32 only, with greedy decoding and a
model-specific single-file Qwen3 weight loader. The desktop chat reconstructs
KV state each turn, and only checks cancellation during token generation.
No GPU, quantization, broad model-format support or production security
hardening is claimed.

The Qwen3 sample model, weights, and tokenizer artifacts are not included in
the repository. Respect the model's upstream license when obtaining them.

## 6. Verification status

| Evidence | Status |
| --- | --- |
| Source code and unit tests present | Confirmed by code inspection |
| Compiler-independent Release assertions | CMake patch committed, requires build verification |
| Windows/MinGW or MSVC test run | Not yet independently executed |
| Matmul benchmark correctness and timings | Benchmark source committed, requires execution |
| Reference Qwen3 logit accuracy | Comparison script exists, results not yet supplied |
| Exact Qwen3 tokenizer token-ID parity | Not yet established |

Update this page with **actual** command outputs, the date, compiler, model
revision, and hardware after running the checks.
