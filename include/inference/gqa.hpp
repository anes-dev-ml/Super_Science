#pragma once

// Grouped-query attention and per-head normalization.
// Included through tensor_inference.hpp, after Tensor is declared.

struct GQAAttentionWeights
{
    std::vector<Tensor> queryWeights;
    std::vector<Tensor> keyWeights;
    std::vector<Tensor> valueWeights;
    Tensor outputWeight;
    Tensor queryNormWeight;
    Tensor keyNormWeight;
};

inline std::size_t getKVHeadIndex(
    std::size_t queryHead,
    std::size_t numQueryHeads,
    std::size_t numKVHeads)
{
    if (numQueryHeads == 0 || numKVHeads == 0)
    {
        throw std::invalid_argument("Query and KV head counts must be positive.");
    }
    if (numQueryHeads % numKVHeads != 0)
    {
        throw std::invalid_argument("Query head count must be divisible by KV head count.");
    }
    if (queryHead >= numQueryHeads)
    {
        throw std::out_of_range("Query head index is out of range.");
    }

    const std::size_t queryHeadsPerKVHead = numQueryHeads / numKVHeads;
    return queryHead / queryHeadsPerKVHead;
}

inline Tensor Tensor::groupedQueryAttention(
    const GQAAttentionWeights &weights,
    std::size_t numQueryHeads,
    std::size_t numKVHeads) const
{
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0)
    {
        throw std::invalid_argument("groupedQueryAttention requires an input tensor with shape [T, D], where T and D are positive.");
    }
    if (numQueryHeads == 0 || numKVHeads == 0 || numQueryHeads % numKVHeads != 0)
    {
        throw std::invalid_argument("Query and KV head counts must be positive, and query heads must be divisible by KV heads.");
    }
    if (weights.queryWeights.size() != numQueryHeads ||
        weights.keyWeights.size() != numKVHeads ||
        weights.valueWeights.size() != numKVHeads)
    {
        throw std::invalid_argument("Projection weight counts must match the query and KV head counts.");
    }

    const std::size_t hiddenDim = shape[1];
    std::size_t queryDim = 0;
    std::size_t valueDim = 0;
    for (const Tensor &queryWeight : weights.queryWeights)
    {
        if (queryWeight.shape.size() != 2 || queryWeight.shape[0] != hiddenDim || queryWeight.shape[1] == 0)
        {
            throw std::invalid_argument("Each query weight must have shape [input dimension, query dimension].");
        }
        if (queryDim == 0)
        {
            queryDim = queryWeight.shape[1];
        }
        else if (queryWeight.shape[1] != queryDim)
        {
            throw std::invalid_argument("All query heads must have the same query dimension.");
        }
    }

    for (std::size_t kvHead = 0; kvHead < numKVHeads; ++kvHead)
    {
        const Tensor &keyWeight = weights.keyWeights[kvHead];
        const Tensor &valueWeight = weights.valueWeights[kvHead];
        if (keyWeight.shape.size() != 2 || keyWeight.shape[0] != hiddenDim || keyWeight.shape[1] != queryDim)
        {
            throw std::invalid_argument("Each key weight must have shape [input dimension, query dimension].");
        }
        if (valueWeight.shape.size() != 2 || valueWeight.shape[0] != hiddenDim || valueWeight.shape[1] == 0)
        {
            throw std::invalid_argument("Each value weight must have shape [input dimension, value dimension].");
        }
        if (kvHead == 0)
        {
            valueDim = valueWeight.shape[1];
        }
        else if (valueWeight.shape[1] != valueDim)
        {
            throw std::invalid_argument("All KV heads must have the same value dimension.");
        }
    }

    if (queryDim % 2 != 0)
    {
        throw std::invalid_argument("RoPE requires an even query/key dimension.");
    }

    if (valueDim == 0 || numQueryHeads > std::numeric_limits<std::size_t>::max() / valueDim)
    {
        throw std::length_error("Concatenated attention dimensions exceed the maximum size.");
    }
    const std::size_t concatenatedDim = numQueryHeads * valueDim;
    if (weights.outputWeight.shape.size() != 2 ||
        weights.outputWeight.shape[0] != concatenatedDim ||
        weights.outputWeight.shape[1] != hiddenDim)
    {
        throw std::invalid_argument("Output weight must have shape [query heads * value dimension, input dimension].");
    }

    std::vector<Tensor> keys;
    std::vector<Tensor> values;
    keys.reserve(numKVHeads);
    values.reserve(numKVHeads);
    for (std::size_t kvHead = 0; kvHead < numKVHeads; ++kvHead)
    {
        keys.push_back(
            normalizeAttentionHead(
                linear(weights.keyWeights[kvHead]),
                weights.keyNormWeight).applyRoPE(0));
        values.push_back(linear(weights.valueWeights[kvHead]));
    }

    Tensor concatenated({shape[0], concatenatedDim});
    for (std::size_t queryHead = 0; queryHead < numQueryHeads; ++queryHead)
    {
        const std::size_t kvHead = getKVHeadIndex(queryHead, numQueryHeads, numKVHeads);
        const Tensor query = normalizeAttentionHead(
            linear(weights.queryWeights[queryHead]),
            weights.queryNormWeight).applyRoPE(0);
        const Tensor headOutput = query.causalAttention(keys[kvHead], values[kvHead]);
        for (std::size_t row = 0; row < shape[0]; ++row)
        {
            for (std::size_t column = 0; column < valueDim; ++column)
            {
                concatenated.at({row, queryHead * valueDim + column}) = headOutput.at({row, column});
            }
        }
    }

    return concatenated.linear(weights.outputWeight);
}

inline Tensor Tensor::cachedGroupedQueryAttentionBuffered(
    const GQAAttentionWeights &weights,
    std::vector<KVCacheBuffer> &caches,
    std::size_t position) const
{
    if (shape.size() != 2 || shape[0] != 1 || shape[1] == 0)
    {
        throw std::invalid_argument("cachedGroupedQueryAttentionBuffered requires an input tensor with shape [1, D], where D is positive.");
    }

    const std::size_t numQueryHeads = weights.queryWeights.size();
    const std::size_t numKVHeads = weights.keyWeights.size();
    if (numQueryHeads == 0 || numKVHeads == 0 || numQueryHeads % numKVHeads != 0)
    {
        throw std::invalid_argument("Query and KV head counts must be positive, and query heads must be divisible by KV heads.");
    }
    if (weights.valueWeights.size() != numKVHeads)
    {
        throw std::invalid_argument("Projection weight counts must match the query and KV head counts.");
    }

    if (caches.size() != numKVHeads)
    {
        throw std::invalid_argument("There must be exactly one KV cache per KV head.");
    }

    const std::size_t hiddenDim = shape[1];
    std::size_t queryDim = 0;
    std::size_t valueDim = 0;
    for (const Tensor &queryWeight : weights.queryWeights)
    {
        if (queryWeight.shape.size() != 2 || queryWeight.shape[0] != hiddenDim || queryWeight.shape[1] == 0)
        {
            throw std::invalid_argument("Each query weight must have shape [input dimension, query dimension].");
        }
        if (queryDim == 0)
        {
            queryDim = queryWeight.shape[1];
        }
        else if (queryWeight.shape[1] != queryDim)
        {
            throw std::invalid_argument("All query heads must have the same query dimension.");
        }
    }

    if (queryDim % 2 != 0)
    {
        throw std::invalid_argument("RoPE requires an even query/key dimension.");
    }

    if (weights.keyNormWeight.shape.size() != 1 || weights.keyNormWeight.shape[0] != queryDim)
    {
        throw std::invalid_argument("Key normalization weight must have shape [query dimension].");
    }
    if (weights.queryNormWeight.shape.size() != 1 || weights.queryNormWeight.shape[0] != queryDim)
    {
        throw std::invalid_argument("Query normalization weight must have shape [query dimension].");
    }

    for (std::size_t kvHead = 0; kvHead < numKVHeads; ++kvHead)
    {
        const Tensor &keyWeight = weights.keyWeights[kvHead];
        const Tensor &valueWeight = weights.valueWeights[kvHead];
        if (keyWeight.shape.size() != 2 || keyWeight.shape[0] != hiddenDim || keyWeight.shape[1] != queryDim)
        {
            throw std::invalid_argument("Each key weight must have shape [input dimension, query dimension].");
        }
        if (valueWeight.shape.size() != 2 || valueWeight.shape[0] != hiddenDim || valueWeight.shape[1] == 0)
        {
            throw std::invalid_argument("Each value weight must have shape [input dimension, value dimension].");
        }
        if (kvHead == 0)
        {
            valueDim = valueWeight.shape[1];
        }
        else if (valueWeight.shape[1] != valueDim)
        {
            throw std::invalid_argument("All KV heads must have the same value dimension.");
        }
    }

    if (valueDim == 0 || numQueryHeads > std::numeric_limits<std::size_t>::max() / valueDim)
    {
        throw std::length_error("Concatenated attention dimensions exceed the maximum size.");
    }
    const std::size_t concatenatedDim = numQueryHeads * valueDim;
    if (weights.outputWeight.shape.size() != 2 ||
        weights.outputWeight.shape[0] != concatenatedDim ||
        weights.outputWeight.shape[1] != hiddenDim)
    {
        throw std::invalid_argument("Output weight must have shape [query heads * value dimension, input dimension].");
    }

    for (const KVCacheBuffer &cache : caches)
    {
        if (cache.getLength() != position)
        {
            throw std::invalid_argument("Each KV cache length must match the requested position.");
        }
        if (cache.getKeyDim() != queryDim || cache.getValueDim() != valueDim)
        {
            throw std::invalid_argument("KV cache dimensions must match the key and value projection dimensions.");
        }
    }

    std::vector<Tensor> newKeys;
    std::vector<Tensor> newValues;

    newKeys.reserve(numKVHeads);
    newValues.reserve(numKVHeads);

    // Phase 1: Compute without modifying caches.
    for (std::size_t kvHead = 0; kvHead < numKVHeads; ++kvHead)
    {
        newKeys.push_back(
            normalizeAttentionHead(
                linear(weights.keyWeights[kvHead]),
                weights.keyNormWeight).applyRoPE(position));

        newValues.push_back(
            linear(weights.valueWeights[kvHead]));
    }

    // Phase 2: Update caches.
    for (std::size_t kvHead = 0; kvHead < numKVHeads; ++kvHead)
    {
        caches[kvHead].append(newKeys[kvHead], newValues[kvHead]);
    }

    Tensor concatenated({1, concatenatedDim});
    for (std::size_t queryHead = 0; queryHead < numQueryHeads; ++queryHead)
    {
        const std::size_t kvHead = getKVHeadIndex(queryHead, numQueryHeads, numKVHeads);
        Tensor Q = normalizeAttentionHead(
            linear(weights.queryWeights[queryHead]),
            weights.queryNormWeight).applyRoPE(position);
        Tensor headOutput = caches[kvHead].incrementalAttention(Q);
        for (std::size_t column = 0; column < valueDim; ++column)
        {
            concatenated.at({0, queryHead * valueDim + column}) = headOutput.at({0, column});
        }
    }

    return concatenated.linear(weights.outputWeight);
}

inline Tensor Tensor::normalizeAttentionHead(
    const Tensor &projected,
    const Tensor &normWeight,
    float epsilon) const
{
    const auto &projectedShape = projected.getShape();
    if (projectedShape.size() != 2 || projectedShape[0] == 0 || projectedShape[1] == 0)
    {
        throw std::invalid_argument("normalizeAttentionHead requires projected input with shape [T, d], where T and d are positive.");
    }
    if (normWeight.getShape().size() != 1 || normWeight.getShape()[0] != projectedShape[1])
    {
        throw std::invalid_argument("normalizeAttentionHead requires normWeight with shape [d].");
    }

    return projected.rmsNorm(normWeight, epsilon);
}
