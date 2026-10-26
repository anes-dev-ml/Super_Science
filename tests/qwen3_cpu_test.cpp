#include "qwen3_cpu.hpp"

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
    std::cout << "Qwen3 CPU token-to-logits, decode cache, and greedy generation PASSED!\n";
}
