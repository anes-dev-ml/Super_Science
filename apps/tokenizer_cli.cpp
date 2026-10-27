#include "tokenizer/qwen3_tokenizer.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>

// Windows command lines are UTF-16; the tokenizer expects UTF-8.
// Using the Unicode OS API avoids depending on the active ANSI code page.
static std::string toUtf8(const wchar_t* text)
{
    const int required = WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, text, -1, nullptr, 0, nullptr, nullptr);
    if (required == 0)
        throw std::runtime_error("Cannot convert Windows argument to UTF-8.");
    std::string result(static_cast<std::size_t>(required), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, -1,
                           result.data(), required, nullptr, nullptr) == 0)
        throw std::runtime_error("Cannot convert Windows argument to UTF-8.");
    result.pop_back(); // Remove the trailing terminator.
    return result;
}
#endif

int main(int argc, char** argv)
{
    try
    {
        std::string path;
        std::string text;
#ifdef _WIN32
        (void)argc;
        (void)argv;
        int wideArgc = 0;
        wchar_t** wideArgv = CommandLineToArgvW(GetCommandLineW(), &wideArgc);
        if (!wideArgv)
            throw std::runtime_error("Cannot read Windows command line.");
        if (wideArgc != 3)
        {
            LocalFree(wideArgv);
            std::cerr << "Usage: tokenizer_cli <tokenizer.json> <text>\n";
            return 2;
        }
        try
        {
            path = toUtf8(wideArgv[1]);
            text = toUtf8(wideArgv[2]);
        }
        catch (...)
        {
            LocalFree(wideArgv);
            throw;
        }
        LocalFree(wideArgv);
#else
        if (argc != 3)
        {
            std::cerr << "Usage: tokenizer_cli <tokenizer.json> <text>\n";
            return 2;
        }
        path = argv[1];
        text = argv[2];
#endif
        Qwen3Tokenizer tokenizer(path);
        const auto ids = tokenizer.encode(text);
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            if (i) std::cout << ' ';
            std::cout << ids[i];
        }
        std::cout << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Tokenization failed: " << error.what() << '\n';
        return 1;
    }
}
