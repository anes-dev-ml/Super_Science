#pragma once

// Transformer stack, generation, RoPE and cached attention implementations.
// Included through tensor_inference.hpp, after Tensor is declared.

inline Tensor Tensor::transformerStack(
    const std::vector<TransformerBlockWeights> &blocks,
    const Tensor &finalNormWeight) const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("transformerStack requires a rank-2 input tensor.");
    }

    Tensor current = *this;
    for (const auto &block : blocks)
    {
        current = current.transformerBlock(
            block.attentionNorm,
            block.heads,
            block.outputWeight,
            block.ffnNorm,
            block.gateWeight,
            block.upWeight,
            block.downWeight);
    }

    return current.rmsNorm(finalNormWeight);
}

inline Tensor Tensor::multiHeadAttention(
    const std::vector<AttentionHeadWeights> &heads,
    const Tensor &Wo) const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("multiHeadAttention requires a rank-2 input tensor.");
    }
    if (heads.empty())
    {
        throw std::invalid_argument("multiHeadAttention requires at least one attention head.");
    }

    std::size_t totalValueDim = 0;
    for (const auto &head : heads)
    {
        if (head.Wq.shape.size() != 2 || head.Wk.shape.size() != 2 || head.Wv.shape.size() != 2)
        {
            throw std::invalid_argument("All head weights must be rank-2 tensors.");
        }
        if (head.Wq.shape[0] != shape[1] || head.Wk.shape[0] != shape[1] || head.Wv.shape[0] != shape[1])
        {
            throw std::invalid_argument("All head weights must have the same input dimension as the input tensor.");
        }
        if (head.Wv.shape[1] > std::numeric_limits<std::size_t>::max() - totalValueDim)
        {
            throw std::length_error("Concatenated value dimensions exceed the maximum size.");
        }
        totalValueDim += head.Wv.shape[1];
    }

    if (Wo.shape.size() != 2 || Wo.shape[0] != totalValueDim)
    {
        throw std::invalid_argument("Output weight input dimension must match the concatenated value dimensions.");
    }
    if (Wo.shape[1] != shape[1])
    {
        throw std::invalid_argument("Output dimension must match the model hidden dimension.");
    }

    Tensor concatenatedValues({shape[0], totalValueDim});
    std::size_t offset = 0;

    for (const auto &head : heads)
    {
        if (head.Wq.shape[1] != head.Wk.shape[1])
        {
            throw std::invalid_argument("Query and key dimensions must match within each head.");
        }

        if (head.Wq.shape[1] % 2 != 0)
        {
            throw std::invalid_argument("RoPE requires an even query/key dimension.");
        }

        Tensor queries = linear(head.Wq).applyRoPE(0);
        Tensor keys = linear(head.Wk).applyRoPE(0);
        Tensor values = linear(head.Wv);
        Tensor attendedValues = queries.causalAttention(keys, values);

        for (std::size_t row = 0; row < shape[0]; ++row)
        {
            for (std::size_t column = 0; column < head.Wv.shape[1]; ++column)
            {
                concatenatedValues.at({row, offset + column}) = attendedValues.at({row, column});
            }
        }
        offset += head.Wv.shape[1];
    }

    Tensor output = concatenatedValues.linear(Wo);
    return output;
}

inline Tensor Tensor::transformerBlock(
    const Tensor &attentionNormWeight,
    const std::vector<AttentionHeadWeights> &heads,
    const Tensor &outputWeight,
    const Tensor &ffnNormWeight,
    const Tensor &gateWeight,
    const Tensor &upWeight,
    const Tensor &downWeight) const
{
    Tensor normedInput = rmsNorm(attentionNormWeight);
    Tensor attentionOutput = normedInput.multiHeadAttention(heads, outputWeight);
    Tensor attentionResidual = add(attentionOutput);
    Tensor normedH = attentionResidual.rmsNorm(ffnNormWeight);
    Tensor feedForward = normedH.swiglu(gateWeight, upWeight, downWeight);
    return attentionResidual.add(feedForward);
}

inline Tensor Tensor::embeddingLookup(
    const std::vector<std::size_t> &tokenIds) const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("embeddingLookup requires a rank-2 embedding table.");
    }

    if (tokenIds.empty())
    {
        throw std::invalid_argument("embeddingLookup requires at least one token ID.");
    }

    Tensor result({tokenIds.size(), shape[1]});

    for (std::size_t i = 0; i < tokenIds.size(); ++i)
    {
        if (tokenIds[i] >= shape[0])
        {
            throw std::out_of_range("Embedding index is out of bounds.");
        }

        for (std::size_t j = 0; j < shape[1]; ++j)
        {
            result.data[i * shape[1] + j] = data[tokenIds[i] * shape[1] + j];
        }
    }

    return result;
}

inline std::size_t Tensor::predictNextToken(
    const std::vector<std::size_t> &tokenIds,
    const std::vector<TransformerBlockWeights> &blocks,
    const Tensor &finalNormWeight) const
{
    Tensor embeddings = embeddingLookup(tokenIds);
    Tensor transformerOutput = embeddings.transformerStack(blocks, finalNormWeight);
    Tensor logits = transformerOutput.lmHead(*this);
    return logits.argmaxLastToken();
}

inline std::vector<std::size_t> Tensor::generate(
    const std::vector<std::size_t> &promptTokens,
    const std::vector<TransformerBlockWeights> &blocks,
    const Tensor &finalNormWeight,
    std::size_t maxNewTokens) const
{
    if (promptTokens.empty())
    {
        throw std::invalid_argument("generate requires at least one prompt token.");
    }

    std::vector<std::size_t> generatedTokens;
    if (promptTokens.size() > generatedTokens.max_size() ||
        maxNewTokens > generatedTokens.max_size() - promptTokens.size())
    {
        throw std::length_error("Requested generation exceeds the maximum token count.");
    }
    generatedTokens.reserve(promptTokens.size() + maxNewTokens);
    generatedTokens.insert(generatedTokens.end(), promptTokens.begin(), promptTokens.end());

    for (std::size_t i = 0; i < maxNewTokens; ++i)
    {
        std::size_t nextToken = predictNextToken(generatedTokens, blocks, finalNormWeight);
        generatedTokens.push_back(nextToken);
    }

    return generatedTokens;
}

inline Tensor Tensor::applyRoPE(std::size_t position) const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("applyRoPE requires a rank-2 tensor.");
    }

    const std::size_t seqLength = shape[0];
    const std::size_t featureDim = shape[1];

    if (featureDim % 2 != 0)
    {
        throw std::invalid_argument("applyRoPE requires an even feature dimension.");
    }

    Tensor result(shape);

    for (std::size_t i = 0; i < seqLength; ++i)
    {
        for (std::size_t j = 0; j < featureDim / 2; ++j)
        {
            float theta = static_cast<float>(position + i) / std::pow(10000.0f, static_cast<float>(j) / (featureDim / 2));
            float cosTheta = std::cos(theta);
            float sinTheta = std::sin(theta);

            float x1 = data[i * featureDim + 2 * j];
            float x2 = data[i * featureDim + 2 * j + 1];

            result.data[i * featureDim + 2 * j] = x1 * cosTheta - x2 * sinTheta;
            result.data[i * featureDim + 2 * j + 1] = x1 * sinTheta + x2 * cosTheta;
        }
    }

    return result;
}

inline Tensor Tensor::appendRows(const Tensor &other) const
{
    if (shape.size() != 2 || other.shape.size() != 2)
    {
        throw std::invalid_argument("appendRows requires rank-2 tensors.");
    }
    if (shape[1] != other.shape[1])
    {
        throw std::invalid_argument("appendRows requires tensors to have the same number of columns.");
    }
    if (other.shape[0] >
        std::numeric_limits<std::size_t>::max() - shape[0])
    {
        throw std::overflow_error("appendRows row count overflow.");
    }

    const std::size_t newRowCount = shape[0] + other.shape[0];
    const std::size_t columnCount = shape[1];

    Tensor result({newRowCount, columnCount});

    for (std::size_t i = 0; i < shape[0]; ++i)
    {
        for (std::size_t j = 0; j < columnCount; ++j)
        {
            result.data[i * columnCount + j] = data[i * columnCount + j];
        }
    }

    for (std::size_t i = 0; i < other.shape[0]; ++i)
    {
        for (std::size_t j = 0; j < columnCount; ++j)
        {
            result.data[(shape[0] + i) * columnCount + j] = other.data[i * columnCount + j];
        }
    }

    return result;
}

inline KVCache Tensor::appendKV(
    const KVCache &cache,
    const Tensor &newKeys,
    const Tensor &newValues) const
{
    if (newKeys.shape.size() != 2 || newValues.shape.size() != 2)
    {
        throw std::invalid_argument("appendKV requires rank-2 tensors for keys and values.");
    }
    if (newKeys.shape[0] != newValues.shape[0])
    {
        throw std::invalid_argument("appendKV requires keys and values to have the same number of rows.");
    }
    if (cache.keys.shape.size() != 2 || cache.values.shape.size() != 2)
    {
        throw std::invalid_argument("appendKV requires rank-2 tensors for cached keys and values.");
    }
    if (cache.keys.shape[0] != cache.values.shape[0])
    {
        throw std::invalid_argument("appendKV requires cached keys and values to have the same number of rows.");
    }
    if (cache.keys.shape[1] != newKeys.shape[1] || cache.values.shape[1] != newValues.shape[1])
    {
        throw std::invalid_argument("appendKV requires matching feature dimensions for cached and new keys/values.");
    }

    return {
        cache.keys.appendRows(newKeys),
        cache.values.appendRows(newValues)};
}

inline Tensor Tensor::incrementalAttention(
    const Tensor &cachedKeys,
    const Tensor &cachedValues) const
{
    if (shape.size() != 2 || cachedKeys.shape.size() != 2 || cachedValues.shape.size() != 2)
    {
        throw std::invalid_argument("incrementalAttention requires rank-2 tensors.");
    }
    if (shape[1] != cachedKeys.shape[1])
    {
        throw std::invalid_argument(
            "Query and key feature dimensions must match.");
    }
    if (cachedKeys.shape[0] != cachedValues.shape[0])
    {
        throw std::invalid_argument("incrementalAttention requires cached keys and values to have the same number of rows.");
    }
    if (shape[0] != 1)
    {
        throw std::invalid_argument(
            "incrementalAttention requires exactly one query token.");
    }

    Tensor scores = attentionScores(cachedKeys);
    Tensor attentionWeights = scores.softmaxLastDim();
    return attentionWeights.matmul(cachedValues);
}

inline CachedAttentionResult Tensor::cachedMultiHeadAttention(
    const std::vector<AttentionHeadWeights> &heads,
    const Tensor &Wo,
    const std::vector<KVCache> &caches,
    std::size_t position) const
{
    if (shape.size() != 2 || shape[0] != 1)
    {
        throw std::invalid_argument("cachedMultiHeadAttention requires an input tensor with shape [1, D].");
    }
    if (heads.empty() || heads.size() != caches.size())
    {
        throw std::invalid_argument("cachedMultiHeadAttention requires a nonempty head collection matching the caches.");
    }
    if (position == 0)
    {
        throw std::invalid_argument("Initialized caches require position to be at least 1.");
    }

    std::size_t totalValueDim = 0;
    for (std::size_t h = 0; h < heads.size(); ++h)
    {
        const auto &head = heads[h];
        const auto &cache = caches[h];
        if (head.Wq.shape.size() != 2 || head.Wk.shape.size() != 2 || head.Wv.shape.size() != 2)
        {
            throw std::invalid_argument("All head weights must be rank-2 tensors.");
        }
        if (head.Wq.shape[0] != shape[1] || head.Wk.shape[0] != shape[1] || head.Wv.shape[0] != shape[1])
        {
            throw std::invalid_argument("All head weights must match the input feature dimension.");
        }
        if (head.Wq.shape[1] != head.Wk.shape[1] || head.Wq.shape[1] % 2 != 0)
        {
            throw std::invalid_argument("Query and key dimensions must match and be even for RoPE.");
        }
        if (cache.keys.shape.size() != 2 || cache.values.shape.size() != 2 ||
            cache.keys.shape[0] != position || cache.values.shape[0] != position)
        {
            throw std::invalid_argument("Each cache must contain exactly position key and value rows.");
        }
        if (cache.keys.shape[1] != head.Wk.shape[1] || cache.values.shape[1] != head.Wv.shape[1])
        {
            throw std::invalid_argument("Cache feature dimensions must match their key and value projections.");
        }
        totalValueDim += head.Wv.shape[1];
    }

    if (Wo.shape.size() != 2 || Wo.shape[0] != totalValueDim)
    {
        throw std::invalid_argument("Output weight input dimension must match the concatenated value dimensions.");
    }
    if (Wo.shape[1] != shape[1])
    {
        throw std::invalid_argument("Output dimension must match the model hidden dimension.");
    }

    Tensor concatenatedValues({1, totalValueDim});
    std::size_t offset = 0;

    std::vector<KVCache> updatedCaches;
    updatedCaches.reserve(heads.size());

    for (std::size_t h = 0; h < heads.size(); ++h)
    {
        const auto &head = heads[h];
        const auto &cache = caches[h];

        Tensor Q = linear(head.Wq);
        Tensor K_new = linear(head.Wk);
        Tensor V_new = linear(head.Wv);
        Q = Q.applyRoPE(position);
        K_new = K_new.applyRoPE(position);

        KVCache updatedCache = appendKV(cache, K_new, V_new);

        Tensor Y = Q.incrementalAttention(
            updatedCache.keys,
            updatedCache.values);

        updatedCaches.push_back(std::move(updatedCache));

        for (std::size_t j = 0; j < head.Wv.shape[1]; ++j)
        {
            concatenatedValues.data[offset + j] = Y.data[j];
        }
        offset += head.Wv.shape[1];
    }

    Tensor output = concatenatedValues.linear(Wo);
    return {output, updatedCaches};
}

inline CachedBlockResult Tensor::cachedTransformerBlock(
    const TransformerBlockWeights &weights,
    const std::vector<KVCache> &caches,
    std::size_t position) const
{
    if (shape.size() != 2 || shape[0] != 1)
    {
        throw std::invalid_argument("cachedTransformerBlock requires an input tensor with shape [1, D].");
    }
    if (weights.heads.size() != caches.size())
    {
        throw std::invalid_argument("cachedTransformerBlock requires a head collection matching the caches.");
    }
    if (weights.heads.empty())
    {
        throw std::invalid_argument("cachedTransformerBlock requires at least one attention head.");
    }

    Tensor normedInput = rmsNorm(weights.attentionNorm);
    CachedAttentionResult attentionResult = normedInput.cachedMultiHeadAttention(weights.heads, weights.outputWeight, caches, position);
    Tensor attentionResidual = this->add(attentionResult.output);
    Tensor normedH = attentionResidual.rmsNorm(weights.ffnNorm);
    Tensor feedForward = normedH.swiglu(weights.gateWeight, weights.upWeight, weights.downWeight);
    Tensor output = attentionResidual.add(feedForward);

    return {
        std::move(output),
        std::move(attentionResult.caches)};
}

inline CachedStackResult Tensor::cachedTransformerStack(
    const std::vector<TransformerBlockWeights> &blocks,
    const std::vector<std::vector<KVCache>> &layerCaches,
    const Tensor &finalNormWeight,
    std::size_t position) const
{
    if (shape.size() != 2 || shape[0] != 1)
    {
        throw std::invalid_argument("cachedTransformerStack requires an input tensor with shape [1, D].");
    }
    if (blocks.size() != layerCaches.size())
    {
        throw std::invalid_argument("cachedTransformerStack requires a block collection matching the layer caches.");
    }
    if (blocks.empty())
    {
        throw std::invalid_argument("cachedTransformerStack requires at least one Transformer block.");
    }

    Tensor current = *this;
    std::vector<std::vector<KVCache>> updatedLayerCaches;
    updatedLayerCaches.reserve(blocks.size());

    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        const auto &block = blocks[i];
        const auto &caches = layerCaches[i];

        CachedBlockResult blockResult = current.cachedTransformerBlock(block, caches, position);
        current = std::move(blockResult.output);
        updatedLayerCaches.push_back(std::move(blockResult.caches));
    }

    Tensor finalOutput = current.rmsNorm(finalNormWeight);

    return {
        std::move(finalOutput),
        std::move(updatedLayerCaches)};
}

inline PrefillAttentionResult Tensor::prefillMultiHeadAttention(
    const std::vector<AttentionHeadWeights> &heads,
    const Tensor &Wo) const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("prefillMultiHeadAttention requires a rank-2 input tensor.");
    }
    if (heads.empty())
    {
        throw std::invalid_argument("prefillMultiHeadAttention requires at least one attention head.");
    }

    std::size_t totalValueDim = 0;
    for (const auto &head : heads)
    {
        if (head.Wq.shape.size() != 2 || head.Wk.shape.size() != 2 || head.Wv.shape.size() != 2)
        {
            throw std::invalid_argument("All head weights must be rank-2 tensors.");
        }
        if (head.Wq.shape[0] != shape[1] || head.Wk.shape[0] != shape[1] || head.Wv.shape[0] != shape[1])
        {
            throw std::invalid_argument("All head weights must have the same input dimension as the input tensor.");
        }
        if (head.Wv.shape[1] > std::numeric_limits<std::size_t>::max() - totalValueDim)
        {
            throw std::length_error("Concatenated value dimensions exceed the maximum size.");
        }

        totalValueDim += head.Wv.shape[1];

        if (head.Wq.shape[1] != head.Wk.shape[1])
        {
            throw std::invalid_argument("Query and key dimensions must match within each head.");
        }
    }

    if (Wo.shape.size() != 2 || Wo.shape[0] != totalValueDim)
    {
        throw std::invalid_argument("Output weight input dimension must match the concatenated value dimensions.");
    }
    if (Wo.shape[1] != shape[1])
    {
        throw std::invalid_argument("Output dimension must match the model hidden dimension.");
    }

    Tensor concatenatedValues({shape[0], totalValueDim});
    std::size_t offset = 0;

    std::vector<KVCache> caches;
    caches.reserve(heads.size());

    for (const auto &head : heads)
    {

        if (head.Wq.shape[1] % 2 != 0)
        {
            throw std::invalid_argument("RoPE requires an even query/key dimension.");
        }

        Tensor queries = linear(head.Wq).applyRoPE(0);
        Tensor keys = linear(head.Wk).applyRoPE(0);
        Tensor values = linear(head.Wv);
        Tensor attendedValues = queries.causalAttention(keys, values);
        const std::size_t valueDim = head.Wv.shape[1];

        for (std::size_t row = 0; row < shape[0]; ++row)
        {
            for (std::size_t column = 0; column < head.Wv.shape[1]; ++column)
            {
                concatenatedValues.data[row * totalValueDim + offset + column] =
                    attendedValues.data[row * valueDim + column];
            }
        }
        offset += head.Wv.shape[1];
        caches.push_back({std::move(keys), std::move(values)});
    }

    Tensor output = concatenatedValues.linear(Wo);
    return {std::move(output), std::move(caches)};
}

inline PrefillBlockResult Tensor::prefillTransformerBlock(
    const TransformerBlockWeights &weights) const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("prefillTransformerBlock requires a rank-2 input tensor.");
    }
    if (weights.heads.empty())
    {
        throw std::invalid_argument("prefillTransformerBlock requires at least one attention head.");
    }

    Tensor normedInput = rmsNorm(weights.attentionNorm);
    PrefillAttentionResult attentionResult = normedInput.prefillMultiHeadAttention(weights.heads, weights.outputWeight);
    Tensor attentionResidual = add(attentionResult.output);
    Tensor normedH = attentionResidual.rmsNorm(weights.ffnNorm);
    Tensor feedForward = normedH.swiglu(weights.gateWeight, weights.upWeight, weights.downWeight);
    Tensor output = attentionResidual.add(feedForward);

    return {
        std::move(output),
        std::move(attentionResult.caches)};
}

inline PrefillStackResult Tensor::prefillTransformerStack(
    const std::vector<TransformerBlockWeights> &blocks,
    const Tensor &finalNormWeight) const
{
    if (shape.size() != 2)
    {
        throw std::invalid_argument("prefillTransformerStack requires a rank-2 input tensor.");
    }
    if (blocks.empty())
    {
        throw std::invalid_argument("prefillTransformerStack requires at least one Transformer block.");
    }

    Tensor current = *this;
    std::vector<std::vector<KVCache>> layerCaches;
    layerCaches.reserve(blocks.size());

    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        PrefillBlockResult blockResult = current.prefillTransformerBlock(blocks[i]);
        current = std::move(blockResult.output);
        layerCaches.push_back(std::move(blockResult.caches));
    }

    Tensor finalOutput = current.rmsNorm(finalNormWeight);

    return {
        std::move(finalOutput),
        std::move(layerCaches)};
}

inline std::vector<std::size_t> Tensor::generateCached(
    const std::vector<std::size_t> &promptTokens,
    const std::vector<TransformerBlockWeights> &blocks,
    const Tensor &finalNormWeight,
    std::size_t maxNewTokens) const
{
    if (promptTokens.empty())
    {
        throw std::invalid_argument("generateCached requires at least one prompt token.");
    }
    if (shape.size() != 2)
    {
        throw std::invalid_argument("generateCached requires a rank-2 embedding table.");
    }

    std::vector<std::size_t> generatedTokens;
    if (promptTokens.size() > generatedTokens.max_size() ||
        maxNewTokens > generatedTokens.max_size() - promptTokens.size())
    {
        throw std::length_error("Requested generation exceeds the maximum token count.");
    }
    generatedTokens.reserve(promptTokens.size() + maxNewTokens);
    generatedTokens.insert(generatedTokens.end(), promptTokens.begin(), promptTokens.end());

    if (maxNewTokens == 0)
    {
        return generatedTokens;
    }

    Tensor promptEmbeddings = embeddingLookup(promptTokens);
    PrefillStackResult prefillResult = promptEmbeddings.prefillTransformerStack(blocks, finalNormWeight);
    std::vector<std::vector<KVCache>> layerCaches = std::move(prefillResult.caches);

    Tensor logits = prefillResult.output.lmHead(*this);
    generatedTokens.push_back(logits.argmaxLastToken());

    for (std::size_t i = 1; i < maxNewTokens; ++i)
    {
        const std::size_t position = generatedTokens.size() - 1;
        Tensor current = embeddingLookup({generatedTokens.back()});
        CachedStackResult cachedResult = current.cachedTransformerStack(
            blocks, layerCaches, finalNormWeight, position);
        layerCaches = std::move(cachedResult.caches);

        logits = cachedResult.output.lmHead(*this);
        generatedTokens.push_back(logits.argmaxLastToken());
    }

    return generatedTokens;
}

inline Tensor KVCacheBuffer::getKeys() const
{
    if (length == 0)
    {
        throw std::logic_error("Cannot retrieve keys from an empty KV cache.");
    }

    Tensor keys({length, keyDim});
    for (std::size_t i = 0; i < length; ++i)
    {
        for (std::size_t j = 0; j < keyDim; ++j)
        {
            keys.at({i, j}) = keyData[i * keyDim + j];
        }
    }
    return keys;
}

inline Tensor KVCacheBuffer::getValues() const
{
    if (length == 0)
    {
        throw std::logic_error("Cannot retrieve values from an empty KV cache.");
    }

    Tensor values({length, valueDim});
    for (std::size_t i = 0; i < length; ++i)
    {
        for (std::size_t j = 0; j < valueDim; ++j)
        {
            values.at({i, j}) = valueData[i * valueDim + j];
        }
    }
    return values;
}

inline Tensor KVCacheBuffer::incrementalAttention(const Tensor &query) const
{
    if (length == 0)
    {
        throw std::logic_error("Cannot perform incremental attention with an empty KV cache.");
    }
    if (query.getShape().size() != 2 || query.getShape()[0] != 1)
    {
        throw std::invalid_argument("incrementalAttention requires a rank-2 tensor with exactly one query token.");
    }
    if (query.getShape()[1] != keyDim)
    {
        throw std::invalid_argument("Query feature dimension must match the key dimension of the KV cache.");
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(keyDim));
    std::vector<float> scores(length);
    float maxScore = -std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < length; ++i)
    {
        float score = 0.0f;
        for (std::size_t j = 0; j < keyDim; ++j)
        {
            score += query.at({0, j}) * keyData[i * keyDim + j];
        }
        scores[i] = score * scale;
        maxScore = std::max(maxScore, scores[i]);
    }

    float scoreSum = 0.0f;
    for (float &score : scores)
    {
        score = std::exp(score - maxScore);
        scoreSum += score;
    }

    Tensor result({1, valueDim});
    for (std::size_t i = 0; i < length; ++i)
    {
        const float weight = scores[i] / scoreSum;
        for (std::size_t j = 0; j < valueDim; ++j)
        {
            result.at({0, j}) += weight * valueData[i * valueDim + j];
        }
    }
    return result;
}

inline Tensor Tensor::cachedMultiHeadAttentionBuffered(
    const std::vector<AttentionHeadWeights> &heads,
    const Tensor &Wo,
    std::vector<KVCacheBuffer> &caches,
    std::size_t position) const
{
    if (shape.size() != 2 || shape[0] != 1)
    {
        throw std::invalid_argument("cachedMultiHeadAttentionBuffered requires an input tensor with shape [1, D].");
    }
    if (heads.empty() || heads.size() != caches.size())
    {
        throw std::invalid_argument("cachedMultiHeadAttentionBuffered requires a nonempty head collection matching the buffers.");
    }
    if (position == 0)
    {
        throw std::invalid_argument("Initialized buffers require position to be at least 1.");
    }

    std::size_t totalValueDim = 0;
    for (std::size_t h = 0; h < heads.size(); ++h)
    {
        const auto &head = heads[h];
        const auto &cache = caches[h];
        if (head.Wq.shape.size() != 2 || head.Wk.shape.size() != 2 || head.Wv.shape.size() != 2)
        {
            throw std::invalid_argument("All head weights must be rank-2 tensors.");
        }
        if (head.Wq.shape[0] != shape[1] || head.Wk.shape[0] != shape[1] || head.Wv.shape[0] != shape[1])
        {
            throw std::invalid_argument("All head weights must match the input feature dimension.");
        }
        if (head.Wq.shape[1] != head.Wk.shape[1] || head.Wq.shape[1] % 2 != 0)
        {
            throw std::invalid_argument("Query and key dimensions must match and be even for RoPE.");
        }
        if (cache.getLength() != position)
        {
            throw std::invalid_argument("Each cache must contain exactly position key and value rows.");
        }
        if (cache.getKeyDim() != head.Wk.shape[1] || cache.getValueDim() != head.Wv.shape[1])
        {
            throw std::invalid_argument("Cache feature dimensions must match their key and value projections.");
        }
        if (head.Wv.shape[1] > std::numeric_limits<std::size_t>::max() - totalValueDim)
        {
            throw std::length_error("Concatenated value dimensions exceed the maximum size.");
        }
        totalValueDim += head.Wv.shape[1];
    }

    if (Wo.shape.size() != 2 || Wo.shape[0] != totalValueDim || Wo.shape[1] != shape[1])
    {
        throw std::invalid_argument("Output weight must have shape [sum of value dimensions, hidden dimension].");
    }

    Tensor concatenatedHeads({1, totalValueDim});
    std::size_t offset = 0;
    for (std::size_t h = 0; h < heads.size(); ++h)
    {
        const auto &head = heads[h];
        Tensor Q = linear(head.Wq).applyRoPE(position);
        Tensor K = linear(head.Wk).applyRoPE(position);
        Tensor V = linear(head.Wv);

        caches[h].append(K, V);
        Tensor headOutput = caches[h].incrementalAttention(Q);
        for (std::size_t j = 0; j < head.Wv.shape[1]; ++j)
        {
            concatenatedHeads.at({0, offset + j}) = headOutput.at({0, j});
        }
        offset += head.Wv.shape[1];
    }

    return concatenatedHeads.linear(Wo);
}

inline Tensor Tensor::cachedTransformerBlockBuffered(
    const TransformerBlockWeights &weights,
    std::vector<KVCacheBuffer> &caches,
    std::size_t position) const
{
    if (shape.size() != 2 || shape[0] != 1)
    {
        throw std::invalid_argument("cachedTransformerBlockBuffered requires an input tensor with shape [1, D].");
    }
    if (weights.heads.size() != caches.size())
    {
        throw std::invalid_argument("cachedTransformerBlockBuffered requires a head collection matching the buffers.");
    }
    if (weights.heads.empty())
    {
        throw std::invalid_argument("cachedTransformerBlockBuffered requires at least one attention head.");
    }
    const std::size_t hiddenDim = shape[1];
    if (weights.attentionNorm.shape != std::vector<std::size_t>{hiddenDim} ||
        weights.ffnNorm.shape != std::vector<std::size_t>{hiddenDim})
    {
        throw std::invalid_argument("Normalization weights must have shape [hidden dimension].");
    }
    if (weights.gateWeight.shape.size() != 2 || weights.upWeight.shape.size() != 2 ||
        weights.downWeight.shape.size() != 2 || weights.gateWeight.shape[0] != hiddenDim ||
        weights.upWeight.shape[0] != hiddenDim || weights.gateWeight.shape[1] != weights.upWeight.shape[1] ||
        weights.downWeight.shape[0] != weights.gateWeight.shape[1] || weights.downWeight.shape[1] != hiddenDim)
    {
        throw std::invalid_argument("FFN weights have incompatible dimensions.");
    }

    Tensor normalized = rmsNorm(weights.attentionNorm);
    Tensor attentionOutput = normalized.cachedMultiHeadAttentionBuffered(
        weights.heads, weights.outputWeight, caches, position);
    Tensor H = this->add(attentionOutput);
    Tensor ffnInput = H.rmsNorm(weights.ffnNorm);
    Tensor ffnOutput = ffnInput.swiglu(
        weights.gateWeight, weights.upWeight, weights.downWeight);
    return H.add(ffnOutput);
}

inline Tensor Tensor::cachedTransformerStackBuffered(
    const std::vector<TransformerBlockWeights> &blocks,
    std::vector<std::vector<KVCacheBuffer>> &layerCaches,
    const Tensor &finalNormWeight,
    std::size_t position) const
{
    if (shape.size() != 2 || shape[0] != 1)
    {
        throw std::invalid_argument("cachedTransformerStackBuffered requires an input tensor with shape [1, D].");
    }
    if (blocks.size() != layerCaches.size())
    {
        throw std::invalid_argument("cachedTransformerStackBuffered requires a block collection matching the layer caches.");
    }
    if (blocks.empty())
    {
        throw std::invalid_argument("cachedTransformerStackBuffered requires at least one Transformer block.");
    }

    const std::size_t hiddenDim = shape[1];
    if (finalNormWeight.shape != std::vector<std::size_t>{hiddenDim})
    {
        throw std::invalid_argument("Final normalization weights must have shape [hidden dimension].");
    }

    for (std::size_t layer = 0; layer < blocks.size(); ++layer)
    {
        if (blocks[layer].heads.size() != layerCaches[layer].size())
        {
            throw std::invalid_argument("Each layer must have one cache per attention head.");
        }
        for (const KVCacheBuffer &cache : layerCaches[layer])
        {
            if (cache.getLength() != position)
            {
                throw std::invalid_argument("Each cache must contain exactly position key and value rows.");
            }
        }
    }

    Tensor current = *this;
    for (std::size_t layer = 0; layer < blocks.size(); ++layer)
    {
        current = current.cachedTransformerBlockBuffered(
            blocks[layer],
            layerCaches[layer],
            position);
    }

    return current.rmsNorm(finalNormWeight);
}

inline std::vector<std::size_t> Tensor::generateBuffered(
    const std::vector<std::size_t> &promptTokens,
    const std::vector<TransformerBlockWeights> &blocks,
    const Tensor &finalNormWeight,
    std::size_t maxNewTokens) const
{
    if (promptTokens.empty())
    {
        throw std::invalid_argument("generateBuffered requires at least one prompt token.");
    }
    if (shape.size() != 2)
    {
        throw std::invalid_argument("generateBuffered requires a rank-2 embedding table.");
    }
    for (std::size_t token : promptTokens)
    {
        if (token >= shape[0])
        {
            throw std::out_of_range("Embedding index is out of bounds.");
        }
    }

    std::vector<std::size_t> generatedTokens;
    if (promptTokens.size() > generatedTokens.max_size() ||
        maxNewTokens > generatedTokens.max_size() - promptTokens.size())
    {
        throw std::length_error("Requested generation exceeds the maximum token count.");
    }
    generatedTokens.reserve(promptTokens.size() + maxNewTokens);
    generatedTokens.insert(generatedTokens.end(), promptTokens.begin(), promptTokens.end());

    if (maxNewTokens == 0)
    {
        return generatedTokens;
    }

    Tensor promptEmbeddings = embeddingLookup(promptTokens);
    PrefillStackResult prefillResult = promptEmbeddings.prefillTransformerStack(blocks, finalNormWeight);
    std::vector<std::vector<KVCacheBuffer>> layerCaches;
    layerCaches.reserve(blocks.size());

    const std::size_t extraCapacity = maxNewTokens - 1;
    for (const auto &layerCache : prefillResult.caches)
    {
        std::vector<KVCacheBuffer> bufferedLayer;
        bufferedLayer.reserve(layerCache.size());
        for (const KVCache &cache : layerCache)
        {
            bufferedLayer.push_back(KVCacheBuffer::fromKVCache(cache, extraCapacity));
        }
        layerCaches.push_back(std::move(bufferedLayer));
    }

    Tensor logits = prefillResult.output.lmHead(*this);
    generatedTokens.push_back(logits.argmaxLastToken());

    for (std::size_t i = 1; i < maxNewTokens; ++i)
    {
        const std::size_t position = generatedTokens.size() - 1;
        Tensor current = embeddingLookup({generatedTokens.back()});
        Tensor output = current.cachedTransformerStackBuffered(
            blocks, layerCaches, finalNormWeight, position);
        logits = output.lmHead(*this);
        generatedTokens.push_back(logits.argmaxLastToken());
    }

    return generatedTokens;
}
