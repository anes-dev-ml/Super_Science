#include "qwen3_cpu.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
    // Small deterministic model to check the complete token-to-logits path.
    const ModelConfig config{4, 1, 2, 1, 2, 8, 4, 1e-6f, 1000000.0f};
    Tensor embeddings({4, 4});
    for (std::size_t i = 0; i < 4; ++i)
        embeddings.at({i, i}) = 1.0f;
    Tensor norm({4});
    Tensor headNorm({2});
    for (std::size_t i = 0; i < 4; ++i)
        norm.at({i}) = 1.0f;
    for (std::size_t i = 0; i < 2; ++i)
        headNorm.at({i}) = 1.0f;

    Qwen3LayerWeights layer{
        norm, norm, headNorm, headNorm,
        Tensor({4, 4}), Tensor({2, 4}), Tensor({2, 4}), Tensor({4, 4}),
        Tensor({8, 4}), Tensor({8, 4}), Tensor({4, 8})};
    Qwen3Cpu model(config, embeddings, {layer}, norm);
    const auto first = model.forwardToken(2);
    assert(first.size() == 4);
    assert(first[2] > first[0] && first[2] > first[1] && first[2] > first[3]);
    assert(model.position() == 1);
    const auto second = model.forwardToken(1);
    assert(second[1] > second[0] && second[1] > second[2] && second[1] > second[3]);
    assert(model.position() == 2);

    const auto tokens = model.generate({2, 1}, 3);
    assert((tokens == std::vector<std::size_t>{2, 1, 1, 1, 1}));
    std::vector<std::size_t> streamed;
    const auto stream = model.generateStreaming({2, 1}, 3, [&](std::size_t id) {
        streamed.push_back(id);
        return true;
    });
    assert((stream == std::vector<std::size_t>{1, 1, 1}));
    assert(streamed == stream);

    std::size_t callbacks = 0;
    const auto stopped = model.generateStreaming({2, 1}, 10, [&](std::size_t) {
        ++callbacks;
        return false;
    });
    assert(callbacks == 1);
    assert(stopped.size() == 1);
    assert(stopped.front() == 1);

    model.reset();
    assert(model.position() == 0);

    bool rejected = false;
    try
    {
        model.forwardToken(4);
    }
    catch (const std::out_of_range &)
    {
        rejected = true;
    }
    assert(rejected);
    // Nonzero Q/K/V and output weights: a preceding token must affect
    // subsequent logits through the causal attention KV cache.
    // The earlier identity-only tests cannot exercise this behavior.
    auto activeLayer = layer;
    activeLayer.query.at({0, 0}) = 1.0f;
    activeLayer.query.at({1, 1}) = 1.0f;
    activeLayer.key.at({0, 0}) = 1.0f;
    activeLayer.key.at({1, 1}) = 1.0f;
    activeLayer.value.at({0, 0}) = 1.0f;
    activeLayer.value.at({1, 1}) = 1.0f;
    activeLayer.output.at({0, 0}) = 0.7f;
    activeLayer.output.at({1, 1}) = 0.7f;

    Qwen3Cpu contextual(config, embeddings, {activeLayer}, norm);
    Qwen3Cpu isolated(config, embeddings, {activeLayer}, norm);
    contextual.forwardToken(0);
    const auto withPreviousToken = contextual.forwardToken(1);
    const auto withoutPreviousToken = isolated.forwardToken(1);
    float maximumContextEffect = 0.0f;
    for (std::size_t i = 0; i < withPreviousToken.size(); ++i)
    {
        assert(std::isfinite(withPreviousToken[i]));
        assert(std::isfinite(withoutPreviousToken[i]));
        maximumContextEffect = std::max(maximumContextEffect,
                                        std::fabs(withPreviousToken[i] - withoutPreviousToken[i]));
    }
    assert(maximumContextEffect > 1e-4f);

    std::cout << "Qwen3 CPU inference, streaming, and nonzero attention context PASSED!\n";
}
