#include "tensor_inference.hpp" // Exercise the public compatibility include.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void require(bool ok, const std::string &message)
{
    if (!ok)
    {
        throw std::runtime_error("FAILED: " + message);
    }
}

template <typename Exception, typename Action>
void expectThrows(Action action, const std::string &message)
{
    try
    {
        action();
    }
    catch (const Exception &)
    {
        return;
    }
    throw std::runtime_error("Expected exception: " + message);
}

void testTensorValidation()
{
    expectThrows<std::invalid_argument>([] { Tensor bad({}); }, "empty shape");
    expectThrows<std::invalid_argument>([] { Tensor bad({2, 0}); }, "zero dimension");
    expectThrows<std::invalid_argument>([] {
        Tensor({2, 3}).matmul(Tensor({4, 2}));
    }, "matmul inner dimensions");

    Tensor a({2, 2});
    expectThrows<std::out_of_range>([&] { a.at({2, 0}); }, "tensor bounds");
    expectThrows<std::invalid_argument>([&] { a.at({0}); }, "tensor rank");

    Tensor logits({2, 3});
    logits.at({0, 0}) = 1000.0f;
    logits.at({0, 1}) = 1001.0f;
    logits.at({0, 2}) = 999.0f;
    logits.at({1, 0}) = -1000.0f;
    logits.at({1, 1}) = -1001.0f;
    logits.at({1, 2}) = -999.0f;
    Tensor probabilities = logits.softmaxLastDim();
    for (std::size_t row = 0; row < 2; ++row)
    {
        float sum = 0.0f;
        for (std::size_t column = 0; column < 3; ++column)
        {
            const float p = probabilities.at({row, column});
            require(std::isfinite(p) && p >= 0.0f && p <= 1.0f, "softmax probability");
            sum += p;
        }
        require(std::fabs(sum - 1.0f) < 1e-6f, "softmax normalization");
    }

    logits.at({0, 0}) = std::numeric_limits<float>::quiet_NaN();
    expectThrows<std::runtime_error>([&] { logits.softmaxLastDim(); }, "NaN logits");
    Tensor rankOne({2});
    rankOne.at({0}) = std::numeric_limits<float>::infinity();
    expectThrows<std::runtime_error>([&] { rankOne.softmax(); }, "infinite logits");
}

void testCacheValidation()
{
    expectThrows<std::invalid_argument>([] { KVCacheBuffer bad(0, 2, 1); }, "zero key size");
    expectThrows<std::invalid_argument>([] { KVCacheBuffer bad(2, 2, 0); }, "zero capacity");
    KVCacheBuffer cache(2, 3, 1);
    expectThrows<std::invalid_argument>([&] {
        cache.append(Tensor({1, 3}), Tensor({1, 3}));
    }, "wrong key width");
    require(cache.getLength() == 0, "invalid append does not mutate cache");
    cache.append(Tensor({1, 2}), Tensor({1, 3}));
    cache.append(Tensor({1, 2}), Tensor({1, 3}));
    require(cache.getLength() == 2 && cache.getCapacity() >= 2, "cache growth");
}

void testModelValidation()
{
    ModelConfig valid{1024, 28, 16, 8, 128, 3072, 151936, 1e-6f, 1e6f};
    validateModelConfig(valid);
    auto invalid = valid;
    invalid.numKVHeads = 3;
    expectThrows<std::invalid_argument>([&] { validateModelConfig(invalid); }, "GQA divisibility");
    invalid = valid;
    invalid.headDim = 127;
    expectThrows<std::invalid_argument>([&] { validateModelConfig(invalid); }, "RoPE head dimension");
    invalid = valid;
    invalid.rmsNormEpsilon = std::numeric_limits<float>::quiet_NaN();
    expectThrows<std::invalid_argument>([&] { validateModelConfig(invalid); }, "finite epsilon");

    SafeTensorInfo info{"x", "F16", {2, 3}, 0, 12};
    validateSafeTensorStorage(info);
    expectThrows<std::invalid_argument>([&] {
        auto bad = info;
        bad.shape = {2, 0};
        validateSafeTensorStorage(bad);
    }, "zero-dimensional storage");
    expectThrows<std::invalid_argument>([&] {
        auto bad = info;
        bad.shape = {};
        validateSafeTensorStorage(bad);
    }, "scalar storage");
    expectThrows<std::invalid_argument>([&] {
        auto bad = info;
        bad.endOffset = 10;
        validateSafeTensorStorage(bad);
    }, "wrong dtype byte count");
    expectThrows<std::invalid_argument>([&] {
        validateSafeTensorRanges({info, SafeTensorInfo{"y", "F16", {1}, 10, 12}}, 12);
    }, "overlapping storage ranges");
    expectThrows<std::invalid_argument>([&] {
        validateSafeTensorRanges({info}, 14);
    }, "uncovered trailing bytes");
    expectThrows<std::invalid_argument>([] { getSafeTensorElementSize("F8"); }, "unsupported dtype");
}
} // namespace

int main()
{
    try
    {
        testTensorValidation();
        testCacheValidation();
        testModelValidation();
        std::cout << "Tensor, cache, model and Safetensors validation tests PASSED!\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
