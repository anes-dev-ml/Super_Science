#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>
#include "tensor.hpp"

template <typename Func>
double benchmarkMs(Func &&function, int repetitions = 10)
{
    using Clock = std::chrono::steady_clock;

    double totalMs = 0.0;

    for (int i = 0; i < repetitions; ++i)
    {
        const auto start = Clock::now();

        function();

        const auto end = Clock::now();

        totalMs += std::chrono::duration<double, std::milli>(
                       end - start)
                       .count();
    }

    return totalMs / repetitions;
}


int main()
{
    KVCacheBuffer cache(2, 3, 2);

    for (std::size_t i = 0; i < 9; ++i)
    {
        Tensor K({1, 2});
        Tensor V({1, 3});

        for (std::size_t j = 0; j < 2; ++j)
        {
            K.at({0, j}) = static_cast<float>(i * 10 + j);
        }

        for (std::size_t j = 0; j < 3; ++j)
        {
            V.at({0, j}) = static_cast<float>(i * 100 + j);
        }

        cache.append(K, V);
    }

    const float *keys = cache.keyDataPtr();
    const float *values = cache.valueDataPtr();

    for (std::size_t i = 0; i < 9; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            assert(keys[i * 2 + j] ==
                   static_cast<float>(i * 10 + j));
        }

        for (std::size_t j = 0; j < 3; ++j)
        {
            assert(values[i * 3 + j] ==
                   static_cast<float>(i * 100 + j));
        }
    }

    std::cout << "KV cache data integrity test PASSED!\n";

    KVCacheBuffer attentionCache(2, 3, 2);

    Tensor K({5, 2});
    Tensor V({5, 3});
    Tensor Q({1, 2});

    for (std::size_t i = 0; i < 5; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            K.at({i, j}) =
                0.1f * static_cast<float>(1 + i * 2 + j);
        }

        for (std::size_t j = 0; j < 3; ++j)
        {
            V.at({i, j}) =
                0.05f * static_cast<float>(1 + i * 3 + j);
        }
    }

    Q.at({0, 0}) = 0.7f;
    Q.at({0, 1}) = -0.3f;

    for (std::size_t i = 0; i < 5; ++i)
    {
        Tensor oneK({1, 2});
        Tensor oneV({1, 3});

        for (std::size_t j = 0; j < 2; ++j)
        {
            oneK.at({0, j}) = K.at({i, j});
        }

        for (std::size_t j = 0; j < 3; ++j)
        {
            oneV.at({0, j}) = V.at({i, j});
        }

        attentionCache.append(oneK, oneV);
    }

    Tensor direct = attentionCache.incrementalAttention(Q);
    Tensor reference = Q.incrementalAttention(K, V);

    float attentionMaxError = 0.0f;

    for (std::size_t j = 0; j < 3; ++j)
    {
        float error = std::abs(
            direct.at({0, j}) - reference.at({0, j}));

        attentionMaxError = std::max(attentionMaxError, error);
    }

    std::cout << "Maximum attention error: "
              << attentionMaxError << '\n';

    assert(std::isfinite(attentionMaxError));
    assert(attentionMaxError < 1e-5f);

    assert(attentionCache.getLength() == 5);
    assert(attentionCache.getCapacity() == 8);

    std::cout << "Direct cached attention test PASSED!\n";

    Tensor input({1, 4});
    for (std::size_t j = 0; j < 4; ++j)
    {
        input.at({0, j}) = static_cast<float>(j + 1);
    }

    Tensor Wq1({4, 2});
    Tensor Wk1({4, 2});
    Tensor Wv1({4, 2});
    Tensor Wq2({4, 2});
    Tensor Wk2({4, 2});
    Tensor Wv2({4, 2});

    Wq1.at({0, 0}) = 1.0f;
    Wq1.at({1, 1}) = 1.0f;
    Wk1.at({0, 0}) = 1.0f;
    Wk1.at({1, 1}) = 1.0f;
    Wv1.at({0, 0}) = 1.0f;
    Wv1.at({1, 1}) = 1.0f;

    Wq2.at({2, 0}) = 1.0f;
    Wq2.at({3, 1}) = 1.0f;
    Wk2.at({2, 0}) = 1.0f;
    Wk2.at({3, 1}) = 1.0f;
    Wv2.at({2, 0}) = 1.0f;
    Wv2.at({3, 1}) = 1.0f;

    Tensor Wo({4, 4});
    for (std::size_t i = 0; i < 4; ++i)
    {
        Wo.at({i, i}) = 1.0f;
    }

    std::vector<AttentionHeadWeights> heads = {
        {Wq1, Wk1, Wv1},
        {Wq2, Wk2, Wv2}};

    std::vector<KVCache> oldCaches;
    std::vector<KVCacheBuffer> newCaches;

    for (std::size_t h = 0; h < heads.size(); ++h)
    {
        Tensor K({2, 2});
        Tensor V({2, 2});

        for (std::size_t i = 0; i < 2; ++i)
        {
            for (std::size_t j = 0; j < 2; ++j)
            {
                K.at({i, j}) =
                    0.1f * static_cast<float>(1 + h * 8 + i * 2 + j);

                V.at({i, j}) =
                    0.05f * static_cast<float>(1 + h * 8 + i * 2 + j);
            }
        }

        oldCaches.push_back({K, V});

        KVCacheBuffer buffer(2, 2, 2);

        for (std::size_t i = 0; i < 2; ++i)
        {
            Tensor oneK({1, 2});
            Tensor oneV({1, 2});

            for (std::size_t j = 0; j < 2; ++j)
            {
                oneK.at({0, j}) = K.at({i, j});
                oneV.at({0, j}) = V.at({i, j});
            }

            buffer.append(oneK, oneV);
        }

        newCaches.push_back(std::move(buffer));
    }

    const std::size_t position = 2;

    auto oldResult = input.cachedMultiHeadAttention(
        heads,
        Wo,
        oldCaches,
        position);

    Tensor newResult = input.cachedMultiHeadAttentionBuffered(
        heads,
        Wo,
        newCaches,
        position);

    float gqaBufferedMaxError = 0.0f;

    for (std::size_t j = 0; j < 4; ++j)
    {
        float error = std::abs(
            oldResult.output.at({0, j}) -
            newResult.at({0, j}));

        gqaBufferedMaxError = std::max(gqaBufferedMaxError, error);
    }

    std::cout << "Buffered MHA maximum error: "
              << gqaBufferedMaxError << '\n';

    assert(std::isfinite(gqaBufferedMaxError));
    assert(gqaBufferedMaxError < 1e-5f);

    for (std::size_t h = 0; h < heads.size(); ++h)
    {
        assert(newCaches[h].getLength() == 3);
        assert(oldResult.caches[h].keys.getShape()[0] == 3);
    }

    std::cout << "Buffered multi-head attention test PASSED!\n";

    Tensor embeddings({8, 4});
    for (std::size_t i = 0; i < 8; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            embeddings.at({i, j}) =
                0.05f * static_cast<float>(1 + i * 4 + j);
        }
    }

    Tensor finalNormWeight({4});
    for (std::size_t j = 0; j < 4; ++j)
    {
        finalNormWeight.at({j}) = 1.0f;
    }

    Tensor headProjection({4, 2});
    Tensor attentionOutput({4, 4});
    Tensor gate({4, 8});
    Tensor up({4, 8});
    Tensor down({8, 4});
    TransformerBlockWeights block{
        finalNormWeight,
        {{headProjection, headProjection, headProjection},
         {headProjection, headProjection, headProjection}},
        attentionOutput,
        finalNormWeight,
        gate,
        up,
        down};
    std::vector<TransformerBlockWeights> blocks = {block};

    const std::vector<std::size_t> prompt = {1, 3, 5};
    const std::size_t newTokens = 5;

    auto full = embeddings.generate(
        prompt, blocks, finalNormWeight, newTokens);

    auto cached = embeddings.generateCached(
        prompt, blocks, finalNormWeight, newTokens);

    auto buffered = embeddings.generateBuffered(
        prompt, blocks, finalNormWeight, newTokens);

    assert(full == cached);
    assert(cached == buffered);

    std::cout << "Full inference: ";
    for (auto token : full)
        std::cout << token << " ";

    std::cout << "\nCached inference: ";
    for (auto token : cached)
        std::cout << token << " ";

    std::cout << "\nBuffered inference: ";
    for (auto token : buffered)
        std::cout << token << " ";

    std::cout << "\nAll three generation paths PASSED!\n";

    const int repetitions = 10;
    const std::size_t benchmarkNewTokens = 50;

    double fullMs = benchmarkMs([&]()
                                {
    auto result = embeddings.generate(
        prompt, blocks, finalNormWeight, benchmarkNewTokens
    );
    volatile std::size_t count = result.size();
    (void)count; }, repetitions);

    double cachedMs = benchmarkMs([&]()
                                  {
    auto result = embeddings.generateCached(
        prompt, blocks, finalNormWeight, benchmarkNewTokens
    );
    volatile std::size_t count = result.size();
    (void)count; }, repetitions);

    double bufferedMs = benchmarkMs([&]()
                                    {
    auto result = embeddings.generateBuffered(
        prompt, blocks, finalNormWeight, benchmarkNewTokens
    );
    volatile std::size_t count = result.size();
    (void)count; }, repetitions);

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Full:     " << fullMs << " ms\n";
    std::cout << "Cached:   " << cachedMs << " ms\n";
    std::cout << "Buffered: " << bufferedMs << " ms\n";

    std::cout << "Buffered vs cached speedup: "
              << cachedMs / bufferedMs << "x\n";

    const std::size_t appendCount = 1000;
    const std::size_t keyWidth = 2;
    const std::size_t valueWidth = 3;

    KVCache originalAppendCache{Tensor({1, keyWidth}), Tensor({1, valueWidth})};
    KVCacheBuffer bufferedAppendCache(keyWidth, valueWidth, 1);
    Tensor initialKey({1, keyWidth});
    Tensor initialValue({1, valueWidth});
    bufferedAppendCache.append(initialKey, initialValue);

    const auto originalAppendStart = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < appendCount; ++i)
    {
        Tensor nextKeys({originalAppendCache.keys.getShape()[0] + 1, keyWidth});
        Tensor nextValues({originalAppendCache.values.getShape()[0] + 1, valueWidth});

        for (std::size_t row = 0; row < originalAppendCache.keys.getShape()[0]; ++row)
            for (std::size_t col = 0; col < keyWidth; ++col)
                nextKeys.at({row, col}) = originalAppendCache.keys.at({row, col});
        for (std::size_t row = 0; row < originalAppendCache.values.getShape()[0]; ++row)
            for (std::size_t col = 0; col < valueWidth; ++col)
                nextValues.at({row, col}) = originalAppendCache.values.at({row, col});

        for (std::size_t col = 0; col < keyWidth; ++col)
            nextKeys.at({originalAppendCache.keys.getShape()[0], col}) =
                static_cast<float>(i + col);
        for (std::size_t col = 0; col < valueWidth; ++col)
            nextValues.at({originalAppendCache.values.getShape()[0], col}) =
                static_cast<float>(i + col);

        originalAppendCache.keys = std::move(nextKeys);
        originalAppendCache.values = std::move(nextValues);
    }
    const auto originalAppendEnd = std::chrono::steady_clock::now();

    const auto bufferedAppendStart = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < appendCount; ++i)
    {
        Tensor oneKey({1, keyWidth});
        Tensor oneValue({1, valueWidth});
        for (std::size_t col = 0; col < keyWidth; ++col)
            oneKey.at({0, col}) = static_cast<float>(i + col);
        for (std::size_t col = 0; col < valueWidth; ++col)
            oneValue.at({0, col}) = static_cast<float>(i + col);
        bufferedAppendCache.append(oneKey, oneValue);
    }
    const auto bufferedAppendEnd = std::chrono::steady_clock::now();

    const double originalAppendMs = std::chrono::duration<double, std::milli>(
                                        originalAppendEnd - originalAppendStart)
                                        .count();
    const double bufferedAppendMs = std::chrono::duration<double, std::milli>(
                                        bufferedAppendEnd - bufferedAppendStart)
                                        .count();

    std::cout << "Original cache append (1000 rows): "
              << originalAppendMs << " ms\n";
    std::cout << "Buffered cache append (1000 rows): "
              << bufferedAppendMs << " ms\n";
    std::cout << "Cache append speedup: "
              << (bufferedAppendMs > 0.0 ? originalAppendMs / bufferedAppendMs : 0.0)
              << "x\n";

    const std::size_t sequenceLength = 4;
    const std::size_t modelDimension = 8;
    const std::size_t numQueryHeads = 4;
    const std::size_t numKVHeads = 2;
    const std::size_t queryKeyDimension = 2;
    const std::size_t valueDimension = 2;

    Tensor gqaInput({sequenceLength, modelDimension});
    for (std::size_t token = 0; token < sequenceLength; ++token)
    {
        for (std::size_t feature = 0; feature < modelDimension; ++feature)
        {
            gqaInput.at({token, feature}) =
                0.1f * static_cast<float>(1 + token * modelDimension + feature);
        }
    }

    GQAAttentionWeights gqaWeights{
        {},
        {},
        {},
        Tensor(std::vector<std::size_t>{numQueryHeads * valueDimension,
                                        modelDimension}),
        Tensor({queryKeyDimension}),
        Tensor({queryKeyDimension})};
    gqaWeights.queryWeights.reserve(numQueryHeads);
    gqaWeights.keyWeights.reserve(numKVHeads);
    gqaWeights.valueWeights.reserve(numKVHeads);

    auto fillDeterministicWeights = [](Tensor &weights, std::size_t seed)
    {
        for (std::size_t row = 0; row < weights.getShape()[0]; ++row)
        {
            for (std::size_t col = 0; col < weights.getShape()[1]; ++col)
            {
                const std::size_t value =
                    1 + (seed + row * 7 + col * 3) % 19;
                weights.at({row, col}) = 0.01f * static_cast<float>(value);
            }
        }
    };

    for (std::size_t h = 0; h < numQueryHeads; ++h)
    {
        Tensor weights({modelDimension, queryKeyDimension});
        fillDeterministicWeights(weights, 1 + h * 2);
        gqaWeights.queryWeights.push_back(std::move(weights));
    }

    for (std::size_t h = 0; h < numKVHeads; ++h)
    {
        Tensor keyWeight({modelDimension, queryKeyDimension});
        Tensor valueWeight({modelDimension, valueDimension});
        fillDeterministicWeights(keyWeight, 2 + h * 3);
        fillDeterministicWeights(valueWeight, 3 + h * 5);
        gqaWeights.keyWeights.push_back(std::move(keyWeight));
        gqaWeights.valueWeights.push_back(std::move(valueWeight));
    }

    fillDeterministicWeights(gqaWeights.outputWeight, 7);
    for (std::size_t feature = 0; feature < queryKeyDimension; ++feature)
    {
        gqaWeights.queryNormWeight.at({feature}) = 1.0f;
        gqaWeights.keyNormWeight.at({feature}) = 1.0f;
    }

    Tensor gqaOutput = gqaInput.groupedQueryAttention(
        gqaWeights,
        numQueryHeads,
        numKVHeads);

    assert(gqaOutput.getShape() ==
           std::vector<std::size_t>({sequenceLength, modelDimension}));
    // GQA normalizes Q and K, so it cannot be compared directly with the
    // unnormalized MHA implementation. Compare full and buffered GQA below.

    std::vector<KVCacheBuffer> gqaCaches;
    gqaCaches.reserve(numKVHeads);

    for (std::size_t h = 0; h < numKVHeads; ++h)
    {
        gqaCaches.emplace_back(
            queryKeyDimension,
            valueDimension,
            sequenceLength);
    }

    float gqaSequenceMaxError = 0.0f;
    for (std::size_t token = 0; token < sequenceLength; ++token)
    {
        Tensor tokenInput({1, modelDimension});
        for (std::size_t feature = 0; feature < modelDimension; ++feature)
            tokenInput.at({0, feature}) = gqaInput.at({token, feature});

        Tensor bufferedOutput = tokenInput.cachedGroupedQueryAttentionBuffered(
            gqaWeights, gqaCaches, token);

        assert(bufferedOutput.getShape() ==
               std::vector<std::size_t>({1, modelDimension}));
        for (std::size_t feature = 0; feature < modelDimension; ++feature)
        {
            gqaSequenceMaxError = std::max(
                gqaSequenceMaxError,
                std::fabs(bufferedOutput.at({0, feature}) -
                          gqaOutput.at({token, feature})));
        }
    }

    assert(std::isfinite(gqaSequenceMaxError));
    assert(gqaSequenceMaxError < 1e-5f);

    for (const auto &cache : gqaCaches)
    {
        assert(cache.getLength() == sequenceLength);
    }

    std::cout << std::scientific
              << std::setprecision(9)
              << "Full/Buffered GQA maximum error: "
              << gqaSequenceMaxError << '\n';

    Tensor projected(std::vector<std::size_t>{2, 2});
    projected.at({0, 0}) = 3.0f;
    projected.at({0, 1}) = 4.0f;
    projected.at({1, 0}) = 3.0f;
    projected.at({1, 1}) = 4.0f;

    Tensor normWeight(std::vector<std::size_t>{2});
    normWeight.at({0}) = 1.0f;
    normWeight.at({1}) = 1.0f;

    Tensor epsilon(std::vector<std::size_t>{1});
    epsilon.at({0}) = 1e-6f;

    Tensor normalized = projected.normalizeAttentionHead(
        projected,
        normWeight,
        epsilon.at({0}));

    const float expected0 = 3.0f / std::sqrt(12.5f + 1e-6f);
    const float expected1 = 4.0f / std::sqrt(12.5f + 1e-6f);

    for (std::size_t row = 0; row < 2; ++row)
    {
        assert(std::fabs(normalized.at({row, 0}) - expected0) < 1e-6f);
        assert(std::fabs(normalized.at({row, 1}) - expected1) < 1e-6f);
    }

    std::cout << "QK normalization test PASSED!\n";
}
