#pragma once

// Compatibility entry point; also safe to include directly.
// The Tensor class must be complete before the inline implementations below.
#include "tensor.hpp"
#include "inference/model_config.hpp"
#include "inference/safetensors.hpp"
#include "inference/kv_cache.hpp"
#include "inference/transformer.hpp"
#include "inference/gqa.hpp"
