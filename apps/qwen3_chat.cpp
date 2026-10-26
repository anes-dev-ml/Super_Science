#include "qwen3_cpu.hpp"
#include "tokenizer/qwen3_tokenizer.hpp"
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: qwen3_chat <model-dir> --prompt \"Hello!\" [--max-new-tokens N] [--raw] [--thinking]\n";
        return 2;
    }
    try
    {
        std::filesystem::path directory = argv[1];
        std::string prompt;
        std::size_t maxTokens = 64;
        bool raw = false, thinking = false;
        for (int i = 2; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--prompt" && i + 1 < argc)
                prompt = argv[++i];
            else if (arg == "--max-new-tokens" && i + 1 < argc)
            {
                auto s = std::string(argv[++i]);
                std::size_t read = 0;
                maxTokens = std::stoull(s, &read);
                if (read != s.size() || maxTokens > 4096)
                    throw std::invalid_argument("Invalid max-new-tokens (maximum 4096)");
            }
            else if (arg == "--raw")
                raw = true;
            else if (arg == "--thinking")
                thinking = true;
            else
                throw std::invalid_argument("Unknown or incomplete argument: " + arg);
        }
        if (prompt.empty())
            throw std::invalid_argument("--prompt must contain text");
        Qwen3Tokenizer tokenizer((directory / "tokenizer.json").string());
        const auto promptIds = raw ? tokenizer.encode(prompt) : tokenizer.chatPrompt(prompt, thinking);
        if (promptIds.empty())
            throw std::invalid_argument("Prompt tokenization produced no tokens");
        std::cerr << "Prompt tokens: " << promptIds.size() << "\nLoading Qwen3 weights...\n";
        Qwen3Cpu model = Qwen3Cpu::fromDirectory(directory);
        auto all = model.generate(promptIds, maxTokens);
        std::vector<std::size_t> generated(all.begin() + static_cast<std::ptrdiff_t>(promptIds.size()), all.end());
        std::cerr << "Generated token IDs: ";
        for (std::size_t id : generated)
        {
            std::cerr << id << ' ';
        }
        std::cerr << '\n';

        std::cout << tokenizer.decode(generated, true) << '\n';
        std::cerr << "Generated " << generated.size() << " tokens\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
