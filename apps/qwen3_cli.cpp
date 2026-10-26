#include "qwen3_cpu.hpp"

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: qwen3_cli <model-directory> <new-token-count|logits> <token-id> [token-id...]\n"
                  << "Requires config.json and model.safetensors in the model directory.\n"
                  << "Tokens must already be encoded using the official Qwen3 tokenizer.\n";
        return 2;
    }

    try
    {
        const bool dumpLogits = std::string(argv[2]) == "logits";
        const std::size_t newTokens = dumpLogits ? 0 : std::stoull(argv[2]);
        std::vector<std::size_t> prompt;
        prompt.reserve(static_cast<std::size_t>(argc - 3));
        for (int i = 3; i < argc; ++i)
            prompt.push_back(std::stoull(argv[i]));

        std::cerr << "Loading model weights (CPU float32)...\n";
        Qwen3Cpu model = Qwen3Cpu::fromDirectory(argv[1]);
        if (dumpLogits)
        {
            std::vector<float> logits;
            for (std::size_t token : prompt)
                logits = model.forwardToken(token);
            std::cout << std::setprecision(9);
            for (std::size_t i = 0; i < logits.size(); ++i)
                std::cout << i << ' ' << logits[i] << '\n';
            return 0;
        }
        const auto generated = model.generate(prompt, newTokens);
        for (std::size_t i = 0; i < generated.size(); ++i)
            std::cout << (i == 0 ? "" : " ") << generated[i];
        std::cout << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Qwen3 inference failed: " << error.what() << '\n';
        return 1;
    }
}
