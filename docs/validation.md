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

**Local CTest status:** build commands were run by the project author, but complete CTest output has not been supplied here. Do not claim a verified 6/6 suite pass until the output or CI run confirms it.

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

**Measured timings:** not yet supplied; no benchmark performance claim is made.
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

**User-reported Qwen3-0.6B comparison results (2026-10-26, local Windows Release build):**

| Token IDs | Maximum absolute error | Mean absolute error | Top token agreement |
| --- | ---: | ---: | --- |
| `1 2 3` | 0.000028014 | 0.000004843 | Yes (16) |
| `1 2 3 4 5 6 7 8 9 10` | 0.000053167 | 0.000008950 | Yes (4999) |
| `100 100 100 100 100 100` | 0.000068665 | 0.000009012 | Yes (14582) |

For a Windows PowerShell 5.1 shell, redirection may create a UTF-16 text file.
The comparison script supports UTF-16 BOM, UTF-8 BOM, and normal UTF-8.

**Autoregressive KV-cache check (user-reported, 2026-10-26):**
Cached C++ greedy generation matched independent Hugging Face `use_cache=False`
full-context generation for 3/3 prompts (12 generated tokens).
The observed sequences were `[3, 4, 3, 4]`,
`[14582, 25, 1019, 40]`, and `[358, 1477, 279, 897]`.
These are bounded test results, not proof of arbitrary context-length equivalence.

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

**User-reported tokenizer parity (2026-10-26):** 15/15 exact token-ID matches
against `transformers.AutoTokenizer` using the local Qwen3-0.6B `tokenizer.json`,
including accented French, Asian scripts, Arabic, emoji, code, and whitespace.

Reproduce locally (after building `tokenizer_cli`):

```powershell
python -m pip install transformers tokenizers
python tools/compare_tokenizer.py models/qwen3-0.6b build-audit/Release/tokenizer_cli.exe
```

This tests plain-text token encoding, **not** official chat-template parity.
The Windows CLI reads Unicode command-line arguments through the wide-character
OS API and converts to UTF-8 before calling the tokenizer.

## 5. Limitations that affect interpretation

The inference engine runs CPU float32 only, with greedy decoding and a
model-specific single-file Qwen3 weight loader. The desktop chat reconstructs
KV state each turn, and only checks cancellation during token generation.
No GPU, quantization, broad model-format support or production security
hardening is claimed.

The Qwen3 sample model, weights, and tokenizer artifacts are not included in
the repository. Respect the model's upstream license when obtaining them.

## 6. CI and evidence status

GitHub Actions workflow `.github/workflows/ci.yml` builds Release configurations
and executes CTest on Windows and Ubuntu. Workflow results must be checked on
GitHub after a push; creating the workflow does not prove that it passes.

| Evidence | Status |
| --- | --- |
| Original C++ inference and tokenizer source | Reviewed in repository |
| Hugging Face numerical logit comparison | 3/3 user-reported passes |
| Official plain-text tokenizer token-ID comparison | 15/15 user-reported passes |
| Qwen3 cached vs full-context greedy generation | 3/3 user-reported passes |
| Complete CTest Release result | Output not yet reviewed |
| Benchmark results | Not yet supplied |
| GitHub Actions Windows and Linux runs | Workflow committed; run results not yet verified |
| Official chat-template equivalence, long-context regression | Not yet independently checked |

Model checkpoint revision, operating system, and compiler details should be
captured with any published reproducibility report. This document intentionally
does not assert broad production correctness or performance guarantees.
