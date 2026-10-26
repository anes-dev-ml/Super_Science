#include "qwen3_cpu.hpp"
#include "tokenizer/qwen3_tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
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
        using Clock = std::chrono::steady_clock;
        const auto loadStart = Clock::now();
        Qwen3Cpu model = Qwen3Cpu::fromDirectory(directory);
        const auto loadEnd = Clock::now();
        const auto start = Clock::now();
        Clock::time_point firstToken;
        bool seenFirst = false;
        const auto generated = model.generateStreaming(promptIds, maxTokens, [&](std::size_t) {
            if (!seenFirst)
            {
                firstToken = Clock::now();
                seenFirst = true;
            }
            return true;
        });
        const auto end = Clock::now();
        std::cerr << "Generated token IDs: ";
        for (std::size_t id : generated)
        {
            std::cerr << id << ' ';
        }
        std::cerr << '\n';

        std::cout << tokenizer.decode(generated, true) << '\n';
        std::cerr << "Generated " << generated.size() << " tokens\n";
        auto ms = [](Clock::duration duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        std::cerr << std::fixed << std::setprecision(1)
                  << "Model load: " << ms(loadEnd - loadStart) << " ms\n"
                  << "Total inference: " << ms(end - start) << " ms\n";
        if (seenFirst)
        {
            const auto firstMs = ms(firstToken - start);
            const auto decodeSeconds = std::chrono::duration<double>(end - firstToken).count();
            std::cerr << "Time to first token (includes prefill): "
                      << firstMs << " ms\n";
            if (generated.size() > 1 && decodeSeconds > 0.0)
                std::cerr << "Subsequent tokens: "
                          << (generated.size() - 1) / decodeSeconds << " tok/s\n";
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
