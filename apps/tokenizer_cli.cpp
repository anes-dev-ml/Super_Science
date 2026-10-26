#include "tokenizer/qwen3_tokenizer.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: tokenizer_cli <tokenizer.json> <text>\n";
        return 2;
    }
    try {
        Qwen3Tokenizer tokenizer(argv[1]);
        const auto ids = tokenizer.encode(argv[2]);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << ids[i];
        }
        std::cout << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Tokenization failed: " << e.what() << '\n';
        return 1;
    }
}
