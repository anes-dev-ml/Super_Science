# Architecture

Super_Science contains two related inference implementations: a general numerical and Transformer path for studying algorithms, and a model-specific Qwen3 path for running local checkpoints.

## Components

| Layer | Source | Responsibility |
| --- | --- | --- |
| Tensor storage | `include/tensor.hpp` | Shape, strides, contiguous float32 storage, arithmetic |
| Transformer experiments | `include/inference/` | Attention, positional encoding, GQA, KV-cache variants |
| Qwen3 runtime | `include/qwen3_cpu.hpp` | Checkpoint loading and token-by-token inference |
| Tokenizer | `include/tokenizer/` | Unicode processing, byte-level BPE, conversation prompts |
| Interfaces | `apps/` | CLI chat, diagnostic CLI, optional desktop chat |

The educational code is not the execution backend for `Qwen3Cpu`. For example, its RoPE implementation uses adjacent-channel rotation with a fixed base, whereas the Qwen3 runtime uses half-vector pairing and the model's configured `rope_theta`.

## Tensor representation

A tensor stores values in a contiguous `std::vector<float>`, accompanied by shape and stride vectors. For a matrix with `C` columns, the offset of element `(r, c)` is `r * C + c`. For higher-rank tensors:

```text
offset = sum(index[axis] * stride[axis])
```

Checked indexing validates shape and bounds. Direct storage access is available for performance-critical loops; callers must validate their own offsets.

The tensor implementation intentionally omits broadcasting, autodifferentiation, and GPU storage.

### Matrix multiplication

The numerical core retains four implementations to compare indexing and memory-access patterns:

| Method | Strategy |
| --- | --- |
| `matmul` | Checked indexing with `i,j,k` loops |
| `matmulDirect` | Direct contiguous indexing |
| `matmulIKJ` | `i,k,j` loops for sequential output writes |
| `matmulBlocked` | Tiled `i,k,j` traversal |

They compute the same operation but have different costs. The optional benchmark verifies outputs before timing each implementation.

## Qwen3 inference pipeline

The model-specific CPU runtime follows the transformer decoder:

```text
token ID
  -> token embedding
  -> repeated decoder layers:
       RMSNorm -> Q/K/V projections -> Q/K normalization -> RoPE
       -> causal grouped-query attention with KV cache
       -> output projection -> residual
       -> RMSNorm -> SwiGLU feed-forward -> residual
  -> final RMSNorm
  -> tied output projection
  -> vocabulary logits
```

Attention uses scaled query–key dot products and a softmax over the current and prior positions. Causality is enforced by processing tokens in order and never inserting future tokens into the cache.

### Key/value cache

Each Qwen3 layer retains the keys and values generated for previous positions. For a new token, the runtime computes its projections and attends to the accumulated cache, avoiding repeated computation of previous decoder states. Intermediate prompt tokens skip the vocabulary projection when logits are not needed.

The CLI uses greedy decoding (`argmax`). The Qt desktop interface shares the same runtime but recomputes the conversation's KV state on every new turn rather than persisting it between turns.

The separate educational Transformer path includes full-sequence, cached, and buffered generation experiments.

## Tokenization and checkpoint loading

The tokenizer reads local `tokenizer.json` data and implements UTF-8 byte-level BPE. The application formats supported Qwen3 chat prompts before passing encoded token IDs to the runtime.

The checkpoint reader parses `config.json`, validates SafeTensors metadata and tensor dimensions, and converts supported F32/F16/BF16 values to float32 in memory. The model-specific loader expects a compatible dense, tied-embedding Qwen3 configuration and a single `model.safetensors` file.

The general SafeTensors format supports more shapes and layouts than this implementation accepts.

## Build targets

| Target | Purpose |
| --- | --- |
| `super_science` | Header-only C++ interface library |
| `qwen3_chat` | Text prompt to generated text |
| `qwen3_cli` | Token IDs and logits inspection |
| `tokenizer_cli` | Native tokenizer diagnostics |
| `*_tests` | Unit and validation executables |
| `matmul_bench` | Optional numerical benchmark |
| `super_science_desktop` | Optional Qt 6 Widgets application |

OpenMP is optional for larger matrix–vector operations. The core has no Python or hosted inference API dependency.

## Design boundaries

This is a CPU float32 implementation for numerical study and supported Qwen3 inference. It does not implement GPU kernels, quantization, sharded checkpoint loading, general model-family dispatch, or persistent KV reuse between desktop chat turns.

For numerical comparisons and the precise extent of tested compatibility, see [Validation](validation.md).
