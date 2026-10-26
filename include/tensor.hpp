#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

class KVCacheBuffer;
struct AttentionHeadWeights;
struct TransformerBlockWeights;
struct KVCache;
struct CachedAttentionResult;
struct CachedBlockResult;
struct CachedStackResult;
struct PrefillAttentionResult;
struct PrefillBlockResult;
struct PrefillStackResult;
struct GQAAttentionWeights;

class Tensor
{
private:
    std::vector<float> data;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;

public:
    explicit Tensor(const std::vector<std::size_t> &dimensions) : shape(dimensions)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Tensor must have at least one dimension.");
        }
        std::size_t count = 1;
        for (std::size_t dimension : shape)
        {
            if (dimension == 0)
            {
                throw std::invalid_argument("Tensor dimensions must be positive.");
            }
            if (count > data.max_size() / dimension)
            {
                throw std::length_error("Tensor dimensions are too large.");
            }
            count *= dimension;
        }

        data.resize(count, 0.0f);
        strides.resize(shape.size());
        strides.back() = 1;

        for (std::size_t i = shape.size() - 1; i > 0; --i)
        {
            strides[i - 1] = strides[i] * shape[i];
        }
    }

    std::size_t getSize() const
    {
        return data.size();
    }

    // Contiguous row-major storage. Useful for model-weight loading and
    // inference kernels where constructing an index vector per float is costly.
    float *dataPtr() noexcept { return data.data(); }
    const float *dataPtr() const noexcept { return data.data(); }

    const std::vector<std::size_t> &getShape() const
    {
        return shape;
    }

    const std::vector<std::size_t> &getStrides() const
    {
        return strides;
    }

    float &at(const std::vector<std::size_t> &coordinates)
    {
        if (coordinates.size() != shape.size())
        {
            throw std::invalid_argument("Index vector must match tensor rank.");
        }

        std::size_t index = 0;
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (coordinates[i] >= shape[i])
            {
                throw std::out_of_range("Index out of bounds.");
            }
            index += coordinates[i] * strides[i];
        }
        return data[index];
    }

    const float &at(const std::vector<std::size_t> &coordinates) const
    {
        if (coordinates.size() != shape.size())
        {
            throw std::invalid_argument("Index vector must match tensor rank.");
        }

        std::size_t index = 0;
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (coordinates[i] >= shape[i])
            {
                throw std::out_of_range("Index out of bounds.");
            }
            index += coordinates[i] * strides[i];
        }
        return data[index];
    }

    Tensor add(const Tensor &other) const
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensor shapes must match.");
        }

        Tensor result(shape);
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }

        return result;
    }

    Tensor multiplyElementwise(const Tensor &other) const
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensor shapes must match.");
        }

        Tensor result(shape);
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] * other.data[i];
        }

        return result;
    }

    Tensor matmul(const Tensor &other) const
    {
        if (shape.size() != 2 || other.shape.size() != 2)
        {
            throw std::invalid_argument("matmul requires rank-2 tensors.");
        }

        std::size_t m = shape[0];
        std::size_t p = other.shape[1];
        std::size_t n = shape[1];

        if (n != other.shape[0])
        {
            throw std::invalid_argument("matmul requires matching inner dimensions.");
        }

        Tensor result({m, p});

        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j < p; ++j)
            {
                float sum = 0.0f;
                for (std::size_t k = 0; k < n; ++k)
                {
                    sum += at({i, k}) * other.at({k, j});
                }
                result.at({i, j}) = sum;
            }
        }

        return result;
    }

    Tensor matmulDirect(const Tensor &other) const
    {
        if (shape.size() != 2 || other.shape.size() != 2)
        {
            throw std::invalid_argument("matmul requires rank-2 tensors.");
        }

        std::size_t m = shape[0];
        std::size_t p = other.shape[1];
        std::size_t n = shape[1];

        if (n != other.shape[0])
        {
            throw std::invalid_argument("matmul requires matching inner dimensions.");
        }

        Tensor result({m, p});

        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j < p; ++j)
            {
                float sum = 0.0f;
                for (std::size_t k = 0; k < n; ++k)
                {
                    sum += data[i * n + k] * other.data[k * p + j];
                }
                result.at({i, j}) = sum;
            }
        }

        return result;
    }

    Tensor matmulIKJ(const Tensor &other) const
    {
        if (shape.size() != 2 || other.shape.size() != 2)
        {
            throw std::invalid_argument("matmul requires rank-2 tensors.");
        }

        const std::size_t m = shape[0];
        const std::size_t p = other.shape[1];
        const std::size_t n = shape[1];

        if (n != other.shape[0])
        {
            throw std::invalid_argument("matmul requires matching inner dimensions.");
        }

        Tensor result({m, p});

        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t k = 0; k < n; ++k)
            {
                const float a = data[i * n + k];
                for (std::size_t j = 0; j < p; ++j)
                {
                    result.data[i * p + j] += a * other.data[k * p + j];
                }
            }
        }

        return result;
    }

    Tensor matmulBlocked(const Tensor &other) const
    {
        constexpr std::size_t BLOCK = 32;
        if (shape.size() != 2 || other.shape.size() != 2)
        {
            throw std::invalid_argument("matmul requires rank-2 tensors.");
        }

        const std::size_t m = shape[0];
        const std::size_t p = other.shape[1];
        const std::size_t n = shape[1];

        if (n != other.shape[0])
        {
            throw std::invalid_argument("matmul requires matching inner dimensions.");
        }

        Tensor result({m, p});
        for (std::size_t ii = 0; ii < m; ii += BLOCK)
        {
            const std::size_t iEnd = std::min(ii + BLOCK, m);
            for (std::size_t kk = 0; kk < n; kk += BLOCK)
            {
                const std::size_t kEnd = std::min(kk + BLOCK, n);
                for (std::size_t jj = 0; jj < p; jj += BLOCK)
                {
                    const std::size_t jEnd = std::min(jj + BLOCK, p);

                    for (std::size_t i = ii; i < iEnd; ++i)
                    {
                        for (std::size_t k = kk; k < kEnd; ++k)
                        {
                            const float a = data[i * n + k];
                            for (std::size_t j = jj; j < jEnd; ++j)
                            {
                                result.data[i * p + j] += a * other.data[k * p + j];
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    Tensor relu() const
    {
        Tensor result(shape);

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = std::max(0.0f, data[i]);
        }

        return result;
    }

    Tensor silu() const
    {
        Tensor result(shape);

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            const float x = data[i];
            result.data[i] = x / (1.0f + std::exp(-x));
        }

        return result;
    }

    Tensor softmax() const
    {
        if (shape.size() != 1)
        {
            throw std::invalid_argument("softmax currently requires a rank-1 tensor.");
        }

        if (data.empty())
        {
            throw std::invalid_argument("softmax requires a non-empty tensor.");
        }

        const float maxValue = *std::max_element(data.begin(), data.end());
        float sum = 0.0f;

        Tensor result(shape);

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            const float x = data[i];
            const float y_i = std::exp(x - maxValue);
            result.data[i] = y_i;
            sum += y_i;
        }

        if (!std::isfinite(sum) || sum <= 0.0f)
        {
            throw std::runtime_error("softmax requires finite logits and a positive denominator.");
        }

        for (std::size_t i = 0; i < result.data.size(); ++i)
        {
            result.data[i] /= sum;
        }

        return result;
    }

    Tensor softmaxLastDim() const
    {
        Tensor result(shape);

        const std::size_t width = shape.back();
        const std::size_t groups = data.size() / width;

        for (std::size_t group = 0; group < groups; ++group)
        {
            const std::size_t start = group * width;

            const float maxValue = *std::max_element(data.begin() + start, data.begin() + start + width);

            float sum = 0.0f;

            for (std::size_t i = start; i < start + width; ++i)
            {
                const float x = data[i];
                const float y_i = std::exp(x - maxValue);
                result.data[i] = y_i;
                sum += y_i;
            }

            if (!std::isfinite(sum) || sum <= 0.0f)
            {
                throw std::runtime_error("softmaxLastDim requires finite logits and a positive denominator.");
            }

            for (std::size_t i = start; i < start + width; ++i)
            {
                result.data[i] /= sum;
            }
        }
        return result;
    }

    Tensor rmsNorm(float epsilon = 1e-5f) const
    {
        if (!std::isfinite(epsilon) || epsilon <= 0.0f)
        {
            throw std::invalid_argument("epsilon must be finite and positive.");
        }

        const std::size_t width = shape.back();

        Tensor result(shape);
        const std::size_t groups = data.size() / width;

        for (std::size_t g = 0; g < groups; ++g)
        {
            const std::size_t start = g * width;

            float sumSquares = 0.0f;
            for (std::size_t i = start; i < start + width; ++i)
            {
                const float x = data[i];
                sumSquares += x * x;
            }

            const float rms = std::sqrt(sumSquares / static_cast<float>(width) + epsilon);
            for (std::size_t i = start; i < start + width; ++i)
            {
                result.data[i] = data[i] / rms;
            }
        }

        return result;
    }

    Tensor rmsNorm(const Tensor &weight, float epsilon = 1e-5f) const
    {

        if (weight.shape.size() != 1 || weight.shape[0] != shape.back())
        {
            throw std::invalid_argument("RMSNorm weight must match the last dimension.");
        }

        Tensor result = rmsNorm(epsilon);

        const std::size_t width = shape.back();

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] *= weight.data[i % width];
        }

        return result;
    }

    Tensor linear(const Tensor &weight, const Tensor &bias) const
    {

        if (weight.shape.size() != 2 || weight.shape[0] != shape.back())
        {
            throw std::invalid_argument("Linear weight must have shape [input_dim, output_dim].");
        }

        if (bias.shape.size() != 1 || bias.shape[0] != weight.shape[1])
        {
            throw std::invalid_argument("Linear bias must match the output dimension.");
        }

        const std::size_t inputDim = shape.back();
        const std::size_t outputDim = weight.shape[1];
        const std::size_t rows = data.size() / inputDim;

        std::vector<std::size_t> outputShape = shape;
        outputShape.back() = outputDim;
        Tensor result(outputShape);

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t k = 0; k < inputDim; ++k)
            {
                const float a = data[i * inputDim + k];
                for (std::size_t j = 0; j < outputDim; ++j)
                {
                    result.data[i * outputDim + j] += a * weight.data[k * outputDim + j];
                }
            }
            for (std::size_t j = 0; j < outputDim; ++j)
            {
                result.data[i * outputDim + j] += bias.data[j];
            }
        }

        return result;
    }

    Tensor linear(const Tensor &weight) const
    {

        if (weight.shape.size() != 2 || weight.shape[0] != shape.back())
        {
            throw std::invalid_argument("Linear weight must have shape [input_dim, output_dim].");
        }

        const std::size_t inputDim = shape.back();
        const std::size_t outputDim = weight.shape[1];
        const std::size_t rows = data.size() / inputDim;

        std::vector<std::size_t> outputShape = shape;
        outputShape.back() = outputDim;
        Tensor result(outputShape);

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t k = 0; k < inputDim; ++k)
            {
                const float a = data[i * inputDim + k];
                for (std::size_t j = 0; j < outputDim; ++j)
                {
                    result.data[i * outputDim + j] += a * weight.data[k * outputDim + j];
                }
            }
        }

        return result;
    }

    Tensor swiglu(const Tensor &gateWeight, const Tensor &upWeight, const Tensor &downWeight) const
    {
        if (gateWeight.shape.size() != 2 || upWeight.shape.size() != 2 || downWeight.shape.size() != 2)
        {
            throw std::invalid_argument("All weights must be rank-2 tensors.");
        }

        if (gateWeight.shape[0] != shape.back() || upWeight.shape[0] != shape.back())
        {
            throw std::invalid_argument("Gate and up weights must match the input dimension.");
        }
        if (gateWeight.shape[1] != upWeight.shape[1])
        {
            throw std::invalid_argument("Gate and up weights must have the same output dimension.");
        }
        if (downWeight.shape[0] != gateWeight.shape[1])
        {
            throw std::invalid_argument("Down weight input dimension must match the gate and up output dimensions.");
        }

        if (downWeight.shape[1] != shape.back())
        {
            throw std::invalid_argument("Down weight output dimension must match input dimension.");
        }

        const Tensor gate = linear(gateWeight).silu();
        const Tensor up = linear(upWeight);
        return gate.multiplyElementwise(up).linear(downWeight);
    }

    Tensor transpose2D() const
    {

        if (shape.size() != 2)
        {
            throw std::invalid_argument("transpose2D requires a rank-2 tensor.");
        }

        const std::size_t rows = shape[0];
        const std::size_t cols = shape[1];

        Tensor result({cols, rows});

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                result.data[j * rows + i] = data[i * cols + j];
            }
        }

        return result;
    }

    Tensor attentionScores(const Tensor &keys) const
    {
        if (shape.size() != 2 || keys.shape.size() != 2)
        {
            throw std::invalid_argument("attentionScores requires rank-2 tensors.");
        }

        if (shape[1] != keys.shape[1])
        {
            throw std::invalid_argument("attentionScores requires matching feature dimensions.");
        }

        const std::size_t featureDim = shape[1];
        const float scale = 1.0f / std::sqrt(static_cast<float>(featureDim));

        const Tensor transposedKeys = keys.transpose2D();
        Tensor result = matmulIKJ(transposedKeys);
        for (float &score : result.data)
        {
            score *= scale;
        }

        return result;
    }

    Tensor causalMask() const
    {

        if (shape.size() != 2)
        {
            throw std::invalid_argument("causalMask requires a rank-2 tensor.");
        }

        if (shape[0] != shape[1])
        {
            throw std::invalid_argument(
                "Causal masking requires a square matrix.");
        }

        const std::size_t rows = shape[0];
        const std::size_t cols = shape[1];

        Tensor result = *this;

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                if (j > i)
                {
                    result.data[i * cols + j] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        return result;
    }

    Tensor causalAttention(const Tensor &keys, const Tensor &values) const
    {
        if (shape.size() != 2 || keys.shape.size() != 2 || values.shape.size() != 2)
        {
            throw std::invalid_argument("causalAttention requires rank-2 tensors.");
        }

        if (shape[1] != keys.shape[1] || keys.shape[0] != values.shape[0])
        {
            throw std::invalid_argument("causalAttention requires matching feature dimensions and sequence lengths.");
        }

        if (shape[0] != keys.shape[0] || keys.shape[0] != values.shape[0])
        {
            throw std::invalid_argument("Q, K and V must have the same sequence length.");
        }

        Tensor scores = attentionScores(keys);
        Tensor maskedScores = scores.causalMask();
        Tensor attentionWeights = maskedScores.softmaxLastDim();
        Tensor result = attentionWeights.matmul(values);

        return result;
    }

    Tensor selfAttention(const Tensor &queryWeight, const Tensor &keyWeight, const Tensor &valueWeight, const Tensor &outputWeight) const
    {

        if (shape.size() != 2)
        {
            throw std::invalid_argument("selfAttention requires a rank-2 input tensor.");
        }

        Tensor Q = linear(queryWeight);
        Tensor K = linear(keyWeight);
        Tensor V = linear(valueWeight);

        Tensor Y = Q.causalAttention(K, V);
        Tensor result = Y.linear(outputWeight);

        return result;
    }

    Tensor concatLastDim(const Tensor &other) const
    {
        if (shape.size() != 2 || other.shape.size() != 2)
        {
            throw std::invalid_argument("concatLastDim requires rank-2 tensors.");
        }
        if (shape[0] != other.shape[0])
        {
            throw std::invalid_argument("concatLastDim requires tensors to have the same number of rows.");
        }

        const std::size_t rows = shape[0];
        const std::size_t firstColumns = shape[1];
        const std::size_t secondColumns = other.shape[1];
        if (secondColumns > std::numeric_limits<std::size_t>::max() - firstColumns)
        {
            throw std::length_error("concatLastDim column count overflow.");
        }
        const std::size_t outputColumns = firstColumns + secondColumns;

        Tensor result({rows, outputColumns});

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < outputColumns; ++j)
            {
                const std::size_t outputIndex = i * outputColumns + j;
                if (j < firstColumns)
                {
                    result.data[outputIndex] = data[i * firstColumns + j];
                }
                else
                {
                    result.data[outputIndex] = other.data[i * secondColumns + (j - firstColumns)];
                }
            }
        }

        return result;
    }

    Tensor twoHeadAttention(const Tensor &Wq1, const Tensor &Wk1, const Tensor &Wv1, const Tensor &Wq2, const Tensor &Wk2, const Tensor &Wv2, const Tensor &Wo) const
    {
        if (shape.size() != 2)
        {
            throw std::invalid_argument("twoHeadAttention requires a rank-2 input tensor.");
        }

        const Tensor *projections[] = {&Wq1, &Wk1, &Wv1, &Wq2, &Wk2, &Wv2};
        for (const Tensor *projection : projections)
        {
            if (projection->shape.size() != 2 || projection->shape[0] != shape[1])
            {
                throw std::invalid_argument("Head projection weights must have shape [input_dim, output_dim].");
            }
        }
        if (Wq1.shape[1] != Wk1.shape[1] || Wq2.shape[1] != Wk2.shape[1])
        {
            throw std::invalid_argument("Query and key dimensions must match within each head.");
        }
        if (Wo.shape.size() != 2 || Wo.shape[0] != Wv1.shape[1] + Wv2.shape[1])
        {
            throw std::invalid_argument("Output weight input dimension must match the concatenated value dimensions.");
        }
        if (Wo.shape[1] != shape[1])
        {
            throw std::invalid_argument("Output dimension must match the model hidden dimension.");
        }

        Tensor Q1 = linear(Wq1);
        Tensor K1 = linear(Wk1);
        Tensor V1 = linear(Wv1);

        Tensor Q2 = linear(Wq2);
        Tensor K2 = linear(Wk2);
        Tensor V2 = linear(Wv2);

        Tensor Y1 = Q1.causalAttention(K1, V1);
        Tensor Y2 = Q2.causalAttention(K2, V2);

        Tensor concatenated = Y1.concatLastDim(Y2);
        Tensor result = concatenated.linear(Wo);

        return result;
    }

    Tensor multiHeadAttention(const std::vector<AttentionHeadWeights> &heads, const Tensor &Wo) const;

    Tensor groupedQueryAttention(
        const GQAAttentionWeights &weights,
        std::size_t numQueryHeads,
        std::size_t numKVHeads) const;

    Tensor transformerBlock(
        const Tensor &attentionNormWeight,
        const std::vector<AttentionHeadWeights> &heads,
        const Tensor &outputWeight,
        const Tensor &ffnNormWeight,
        const Tensor &gateWeight,
        const Tensor &upWeight,
        const Tensor &downWeight) const;

    Tensor transformerStack(
        const std::vector<TransformerBlockWeights> &blocks,
        const Tensor &finalNormWeight) const;

    Tensor embeddingLookup(
        const std::vector<std::size_t> &tokenIds) const;

    Tensor lmHead(const Tensor &embeddingWeight) const
    {
        if (embeddingWeight.shape.size() != 2 || embeddingWeight.shape[1] != shape[1])
        {
            throw std::invalid_argument("lmHead requires a rank-2 embedding weight with a matching hidden dimension.");
        }

        Tensor transposedEmbedding = embeddingWeight.transpose2D();
        return linear(transposedEmbedding);
    }

    std::size_t argmaxLastToken() const
    {
        if (shape.size() != 2)
        {
            throw std::invalid_argument("argmaxLastToken requires a rank-2 tensor.");
        }
        if (shape[0] == 0 || shape[1] == 0)
        {
            throw std::invalid_argument("argmaxLastToken requires non-empty sequence and vocabulary dimensions.");
        }

        const std::size_t lastRowIndex = shape[0] - 1;
        const std::size_t lastRowStart = lastRowIndex * shape[1];
        std::size_t maxIndex = 0;
        float maxLogit = data[lastRowStart];
        for (std::size_t i = 1; i < shape[1]; ++i)
        {
            const float logit = data[lastRowStart + i];
            if (logit > maxLogit)
            {
                maxLogit = logit;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    std::size_t predictNextToken(
        const std::vector<std::size_t> &tokenIds,
        const std::vector<TransformerBlockWeights> &blocks,
        const Tensor &finalNormWeight) const;

    std::vector<std::size_t> generate(
        const std::vector<std::size_t> &promptTokens,
        const std::vector<TransformerBlockWeights> &blocks,
        const Tensor &finalNormWeight,
        std::size_t maxNewTokens) const;

    Tensor applyRoPE(std::size_t position) const;

    Tensor appendRows(const Tensor &other) const;

    KVCache appendKV(
        const KVCache &cache,
        const Tensor &newKeys,
        const Tensor &newValues) const;

    Tensor incrementalAttention(
        const Tensor &cachedKeys,
        const Tensor &cachedValues) const;

    CachedAttentionResult cachedMultiHeadAttention(
        const std::vector<AttentionHeadWeights> &heads,
        const Tensor &Wo,
        const std::vector<KVCache> &caches,
        std::size_t position) const;

    CachedBlockResult cachedTransformerBlock(
        const TransformerBlockWeights &weights,
        const std::vector<KVCache> &caches,
        std::size_t position) const;

    CachedStackResult cachedTransformerStack(
        const std::vector<TransformerBlockWeights> &blocks,
        const std::vector<std::vector<KVCache>> &layerCaches,
        const Tensor &finalNormWeight,
        std::size_t position) const;

    PrefillAttentionResult prefillMultiHeadAttention(
        const std::vector<AttentionHeadWeights> &heads,
        const Tensor &Wo) const;

    PrefillBlockResult prefillTransformerBlock(
        const TransformerBlockWeights &weights) const;

    PrefillStackResult prefillTransformerStack(
        const std::vector<TransformerBlockWeights> &blocks,
        const Tensor &finalNormWeight) const;

    std::vector<std::size_t> generateCached(
        const std::vector<std::size_t> &promptTokens,
        const std::vector<TransformerBlockWeights> &blocks,
        const Tensor &finalNormWeight,
        std::size_t maxNewTokens) const;

    Tensor cachedMultiHeadAttentionBuffered(
        const std::vector<AttentionHeadWeights> &heads,
        const Tensor &Wo,
        std::vector<KVCacheBuffer> &caches,
        std::size_t position) const;

    Tensor cachedTransformerBlockBuffered(
        const TransformerBlockWeights &weights,
        std::vector<KVCacheBuffer> &caches,
        std::size_t position) const;

    Tensor cachedTransformerStackBuffered(
        const std::vector<TransformerBlockWeights> &blocks,
        std::vector<std::vector<KVCacheBuffer>> &layerCaches,
        const Tensor &finalNormWeight,
        std::size_t position) const;

    std::vector<std::size_t> generateBuffered(
        const std::vector<std::size_t> &promptTokens,
        const std::vector<TransformerBlockWeights> &blocks,
        const Tensor &finalNormWeight,
        std::size_t maxNewTokens) const;

    Tensor cachedGroupedQueryAttentionBuffered(
        const GQAAttentionWeights &weights,
        std::vector<KVCacheBuffer> &caches,
        std::size_t position) const;

    Tensor normalizeAttentionHead(
        const Tensor &projected,
        const Tensor &normWeight,
        float epsilon = 1e-6f) const;

};

// Deferred definitions and inference-specific supporting types.
#include "tensor_inference.hpp"
