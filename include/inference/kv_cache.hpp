#pragma once

// KV cache storage, attention/block weight types and inference results.
// Included through tensor_inference.hpp, after Tensor is declared.

class KVCacheBuffer
{
private:
    std::size_t length;
    std::size_t capacity;
    std::size_t keyDim;
    std::size_t valueDim;

    std::vector<float> keyData;
    std::vector<float> valueData;

    void grow(std::size_t newCapacity)
    {
        if (newCapacity <= capacity)
        {
            throw std::invalid_argument("New capacity must be greater than current capacity.");
        }

        const std::size_t maxSize = std::numeric_limits<std::size_t>::max();
        if (newCapacity > maxSize / keyDim)
        {
            throw std::length_error("Key storage size overflow.");
        }
        if (newCapacity > maxSize / valueDim)
        {
            throw std::length_error("Value storage size overflow.");
        }

        const std::size_t newKeyCount = newCapacity * keyDim;
        const std::size_t newValueCount = newCapacity * valueDim;

        if (newKeyCount > keyData.max_size())
        {
            throw std::length_error("Key storage exceeds vector max_size.");
        }
        if (newValueCount > valueData.max_size())
        {
            throw std::length_error("Value storage exceeds vector max_size.");
        }

        std::vector<float> newKeyData(newKeyCount);
        std::vector<float> newValueData(newValueCount);

        std::copy_n(keyData.begin(), length * keyDim, newKeyData.begin());
        std::copy_n(valueData.begin(), length * valueDim, newValueData.begin());

        keyData.swap(newKeyData);
        valueData.swap(newValueData);
        capacity = newCapacity;
    }

    void grow()
    {
        const std::size_t maxCapacity = std::min(
            keyData.max_size() / keyDim,
            valueData.max_size() / valueDim);

        if (capacity >= maxCapacity)
        {
            throw std::length_error("KV cache cannot grow beyond vector max_size.");
        }

        const std::size_t newCapacity = capacity > maxCapacity / 2
                                            ? maxCapacity
                                            : capacity * 2;
        grow(newCapacity);
    }

public:
    KVCacheBuffer(
        std::size_t keyDimension,
        std::size_t valueDimension,
        std::size_t initialCapacity)
        : length(0),
          capacity(initialCapacity),
          keyDim(keyDimension),
          valueDim(valueDimension)
    {
        if (keyDimension == 0 || valueDimension == 0 || initialCapacity == 0)
        {
            throw std::invalid_argument("KV cache dimensions and capacity must be positive.");
        }

        const std::size_t maxSize = std::numeric_limits<std::size_t>::max();
        if (initialCapacity > maxSize / keyDimension)
        {
            throw std::length_error("Key storage size overflow.");
        }
        if (initialCapacity > maxSize / valueDimension)
        {
            throw std::length_error("Value storage size overflow.");
        }

        const std::size_t keyCount = initialCapacity * keyDimension;
        const std::size_t valueCount = initialCapacity * valueDimension;
        if (keyCount > keyData.max_size())
        {
            throw std::length_error("Key storage exceeds vector max_size.");
        }
        if (valueCount > valueData.max_size())
        {
            throw std::length_error("Value storage exceeds vector max_size.");
        }

        keyData.resize(keyCount);
        valueData.resize(valueCount);
    }

    std::size_t getLength() const
    {
        return length;
    }

    std::size_t getCapacity() const
    {
        return capacity;
    }

    std::size_t getKeyDim() const
    {
        return keyDim;
    }

    std::size_t getValueDim() const
    {
        return valueDim;
    }

    void append(const Tensor &newKeys, const Tensor &newValues)
    {

        if (newKeys.getShape().size() != 2 || newValues.getShape().size() != 2)
        {
            throw std::invalid_argument("New keys and values must be rank-2 tensors.");
        }

        if (newKeys.getShape()[0] != 1 || newValues.getShape()[0] != 1)
        {
            throw std::invalid_argument("New keys and values must each have exactly one row.");
        }

        if (newKeys.getShape()[1] != keyDim || newValues.getShape()[1] != valueDim)
        {
            throw std::invalid_argument("New keys and values must match the cache dimensions.");
        }

        if (length == capacity)
        {
            grow();
        }

        for (std::size_t j = 0; j < keyDim; ++j)
        {
            keyData[length * keyDim + j] = newKeys.at({0, j});
        }
        for (std::size_t j = 0; j < valueDim; ++j)
        {
            valueData[length * valueDim + j] = newValues.at({0, j});
        }

        ++length;
    }

    Tensor getKeys() const;

    Tensor getValues() const;

    const float *keyDataPtr() const
    {
        return keyData.data();
    }

    const float *valueDataPtr() const
    {
        return valueData.data();
    }

    Tensor incrementalAttention(const Tensor &query) const;

    static KVCacheBuffer fromKVCache(
        const KVCache &source,
        std::size_t extraCapacity = 0);
};

struct KVCache
{
    Tensor keys;
    Tensor values;
};

inline KVCacheBuffer KVCacheBuffer::fromKVCache(
    const KVCache &source,
    std::size_t extraCapacity)
{
    const auto &keyShape = source.keys.getShape();
    const auto &valueShape = source.values.getShape();

    if (keyShape.size() != 2 || valueShape.size() != 2)
    {
        throw std::invalid_argument("KV cache keys and values must be rank-2 tensors.");
    }
    if (keyShape[0] != valueShape[0])
    {
        throw std::invalid_argument("KV cache keys and values must have the same number of rows.");
    }
    if (keyShape[1] == 0 || valueShape[1] == 0)
    {
        throw std::invalid_argument("KV cache key and value dimensions must be positive.");
    }
    if (keyShape[0] == 0)
    {
        throw std::invalid_argument("Cannot convert an empty KV cache.");
    }
    if (extraCapacity > std::numeric_limits<std::size_t>::max() - keyShape[0])
    {
        throw std::length_error("KV cache capacity overflow.");
    }

    const std::size_t capacity = keyShape[0] + extraCapacity;
    KVCacheBuffer result(keyShape[1], valueShape[1], capacity);
    for (std::size_t i = 0; i < keyShape[0]; ++i)
    {
        for (std::size_t j = 0; j < keyShape[1]; ++j)
        {
            result.keyData[i * keyShape[1] + j] = source.keys.at({i, j});
        }
        for (std::size_t j = 0; j < valueShape[1]; ++j)
        {
            result.valueData[i * valueShape[1] + j] = source.values.at({i, j});
        }
    }
    result.length = keyShape[0];
    return result;
}

struct AttentionHeadWeights
{
    Tensor Wq;
    Tensor Wk;
    Tensor Wv;
};

struct TransformerBlockWeights
{
    Tensor attentionNorm;
    std::vector<AttentionHeadWeights> heads;
    Tensor outputWeight;

    Tensor ffnNorm;
    Tensor gateWeight;
    Tensor upWeight;
    Tensor downWeight;
};

struct CachedAttentionResult
{
    Tensor output;
    std::vector<KVCache> caches;
};

struct CachedBlockResult
{
    Tensor output;
    std::vector<KVCache> caches;
};

struct CachedStackResult
{
    Tensor output;
    std::vector<std::vector<KVCache>> caches;
};

struct PrefillAttentionResult
{
    Tensor output;
    std::vector<KVCache> caches;
};

struct PrefillBlockResult
{
    Tensor output;
    std::vector<KVCache> caches;
};

struct PrefillStackResult
{
    Tensor output;
    std::vector<std::vector<KVCache>> caches;
};
