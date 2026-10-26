#pragma once
#include "json.hpp"
#include "tokenizer/unicode_ranges.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Self-contained Qwen3 byte-level BPE tokenizer. Uses only the C++17 standard library.
class Qwen3Tokenizer
{
    struct Unit
    {
        char32_t cp;
        std::string raw;
    };
    std::unordered_map<std::string, std::size_t> vocab_;
    std::unordered_map<std::string, std::size_t> mergeRank_;
    std::unordered_map<std::string, std::size_t> special_;
    std::vector<std::string> idToToken_;
    std::unordered_set<std::size_t> skipOnDecode_;
    std::array<std::string, 256> byteToSymbol_{};
    std::unordered_map<char32_t, unsigned char> symbolToByte_;

    static std::string utf8(char32_t x)
    {
        std::string s;
        if (x <= 0x7f)
            s += char(x);
        else if (x <= 0x7ff)
        {
            s += char(0xc0 | (x >> 6));
            s += char(0x80 | (x & 63));
        }
        else if (x <= 0xffff)
        {
            s += char(0xe0 | (x >> 12));
            s += char(0x80 | ((x >> 6) & 63));
            s += char(0x80 | (x & 63));
        }
        else
        {
            s += char(0xf0 | (x >> 18));
            s += char(0x80 | ((x >> 12) & 63));
            s += char(0x80 | ((x >> 6) & 63));
            s += char(0x80 | (x & 63));
        }
        return s;
    }
    static std::vector<Unit> units(const std::string &s)
    {
        std::vector<Unit> out;
        for (std::size_t i = 0; i < s.size();)
        {
            auto b = static_cast<unsigned char>(s[i]);
            std::size_t n = b < 0x80 ? 1 : (b >= 0xc2 && b <= 0xdf ? 2 : (b >= 0xe0 && b <= 0xef ? 3 : (b >= 0xf0 && b <= 0xf4 ? 4 : 0)));
            if (n == 0 || i + n > s.size())
                throw std::invalid_argument("Invalid UTF-8 input");
            char32_t cp = b & (n == 1 ? 0x7f : (n == 2 ? 0x1f : (n == 3 ? 0x0f : 7)));
            for (std::size_t j = 1; j < n; ++j)
            {
                auto c = static_cast<unsigned char>(s[i + j]);
                if ((c & 0xc0) != 0x80)
                    throw std::invalid_argument("Invalid UTF-8 continuation");
                cp = (cp << 6) | (c & 63);
            }
            if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) || (n == 4 && cp < 0x10000) || (cp >= 0xd800 && cp <= 0xdfff) || cp > 0x10ffff)
                throw std::invalid_argument("Invalid UTF-8 codepoint");
            out.push_back({cp, s.substr(i, n)});
            i += n;
        }
        return out;
    }
    static bool letter(char32_t c) { return qwen_unicode::in(c, qwen_unicode::letter); }
    static bool mark(char32_t c) { return qwen_unicode::in(c, qwen_unicode::mark); }
    static bool number(char32_t c) { return qwen_unicode::in(c, qwen_unicode::number); }
    static bool space(char32_t c) { return qwen_unicode::in(c, qwen_unicode::space); }
    static bool newline(char32_t c) { return c == '\r' || c == '\n'; }
    static bool symbol(char32_t c) { return !space(c) && !letter(c) && !mark(c) && !number(c); }
    static std::vector<std::string> split(const std::string &text)
    {
        const auto u = units(text);
        std::vector<std::string> parts;
        std::size_t i = 0;
        auto emit = [&](std::size_t a, std::size_t b)
        {std::string s;for(auto j=a;j<b;++j)s+=u[j].raw;parts.push_back(std::move(s));i=b; };
        while (i < u.size())
        {
            // First alternative: case-insensitive contractions (ASCII apostrophe only).
            if (u[i].cp == '\'' && i + 1 < u.size())
            {
                const std::string contractions[] = {"s", "t", "re", "ve", "m", "ll", "d"};
                bool matched = false;
                for (const auto &word : contractions)
                {
                    if (i + word.size() >= u.size())
                        continue;
                    bool ok = true;
                    for (std::size_t j = 0; j < word.size(); ++j)
                    {
                        auto c = u[i + 1 + j].cp;
                        if (c >= 'A' && c <= 'Z')
                            c += 32;
                        if (c != static_cast<unsigned char>(word[j]))
                        {
                            ok = false;
                            break;
                        }
                    }
                    if (ok)
                    {
                        emit(i, i + word.size() + 1);
                        matched = true;
                        break;
                    }
                }
                if (matched)
                    continue;
            }
            // Optional single non-letter/number (except CR/LF), followed by letters/marks.
            std::size_t start = i, j = i;
            if (!newline(u[j].cp) && !letter(u[j].cp) && !number(u[j].cp))
                ++j;
            if (j < u.size() && (letter(u[j].cp) || mark(u[j].cp)))
            {
                while (j < u.size() && (letter(u[j].cp) || mark(u[j].cp)))
                    ++j;
                emit(start, j);
                continue;
            }
            if (number(u[i].cp))
            {
                emit(i, i + 1);
                continue;
            }
            // Optional *ASCII space*, then punctuation/symbols, then CR/LF.
            j = i;
            if (u[j].cp == ' ')
                ++j;
            std::size_t symStart = j;
            while (j < u.size() && symbol(u[j].cp) && !newline(u[j].cp))
                ++j;
            if (j > symStart)
            {
                while (j < u.size() && newline(u[j].cp))
                    ++j;
                emit(i, j);
                continue;
            }
            // Whitespace* followed by CR/LF; greedy, include the newline run.
            j = i;
            while (j < u.size() && space(u[j].cp) && !newline(u[j].cp))
                ++j;
            if (j < u.size() && newline(u[j].cp))
            {
                while (j < u.size() && newline(u[j].cp))
                    ++j;
                emit(i, j);
                continue;
            }
            // Whitespace+(?!\S) consumes all whitespace except the final one before nonspace.
            if (space(u[i].cp))
            {
                j = i;
                while (j < u.size() && space(u[j].cp))
                    ++j;
                if (j < u.size() && j - i >= 2)
                {
                    emit(i, j - 1);
                    continue;
                }
                emit(i, j);
                continue;
            }
            // Remaining uncommon Unicode codepoints.
            emit(i, i + 1);
        }
        return parts;
    }
    void setupAlphabet()
    {
        bool taken[256] = {};
        for (int b = 33; b <= 126; ++b)
            taken[b] = true;
        for (int b = 161; b <= 172; ++b)
            taken[b] = true;
        for (int b = 174; b <= 255; ++b)
            taken[b] = true;
        int extra = 0;
        for (int b = 0; b < 256; ++b)
        {
            char32_t cp = taken[b] ? static_cast<char32_t>(b) : static_cast<char32_t>(256 + extra++);
            byteToSymbol_[b] = utf8(cp);
            symbolToByte_[cp] = static_cast<unsigned char>(b);
        }
    }
    std::vector<std::size_t> encodePiece(const std::string &part) const
    {
        std::vector<std::string> symbols;
        symbols.reserve(part.size());
        for (unsigned char b : part)
            symbols.push_back(byteToSymbol_[b]);
        while (symbols.size() > 1)
        {
            std::size_t best = std::numeric_limits<std::size_t>::max(), idx = symbols.size();
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i)
            {
                auto it = mergeRank_.find(symbols[i] + symbols[i + 1]);
                if (it != mergeRank_.end() && it->second < best)
                {
                    best = it->second;
                    idx = i;
                }
            }
            if (idx == symbols.size())
                break;
            symbols[idx] += symbols[idx + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(idx + 1));
        }
        std::vector<std::size_t> ids;
        for (const auto &s : symbols)
        {
            auto it = vocab_.find(s);
            if (it == vocab_.end())
                throw std::runtime_error("BPE produced unknown vocabulary token");
            ids.push_back(it->second);
        }
        return ids;
    }

public:
    explicit Qwen3Tokenizer(const std::string &file)
    {
        setupAlphabet();
        std::ifstream in(file);
        if (!in)
            throw std::runtime_error("Cannot open tokenizer.json: " + file);
        nlohmann::json root;
        in >> root;
        if (!root.contains("model") || root.at("model").value("type", std::string()) != "BPE")
            throw std::invalid_argument("Expected a BPE tokenizer.json");
        const auto &model = root.at("model");
        for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it)
        {
            auto id = it.value().get<std::size_t>();
            vocab_[it.key()] = id;
            if (id >= idToToken_.size())
                idToToken_.resize(id + 1);
            idToToken_[id] = it.key();
        }
        std::size_t rank = 0;
        for (const auto &merge : model.at("merges"))
        {
            std::string a, b;
            if (merge.is_string())
            {
                auto line = merge.get<std::string>();
                auto sep = line.find(' ');
                if (sep == std::string::npos)
                    throw std::invalid_argument("Invalid BPE merge");
                a = line.substr(0, sep);
                b = line.substr(sep + 1);
            }
            else if (merge.is_array() && merge.size() == 2)
            {
                a = merge[0].get<std::string>();
                b = merge[1].get<std::string>();
            }
            else
                throw std::invalid_argument("Invalid BPE merge format");
            mergeRank_.emplace(a + b, rank++);
        }
        if (root.contains("added_tokens"))
            for (const auto &t : root.at("added_tokens"))
            {
                const auto content = t.at("content").get<std::string>();
                const auto id = t.at("id").get<std::size_t>();
                const bool isSpecial = t.value("special", false);
                // Added tokens are valid vocabulary entries even if special=false.
                // Qwen3 uses that flag for its <think> and </think> markers.
                if (id >= idToToken_.size())
                    idToToken_.resize(id + 1);
                idToToken_[id] = content;
                if (isSpecial || content == "<think>" || content == "</think>")
                    skipOnDecode_.insert(id);
                // Permit explicit insertion of any added token in a chat template.
                special_[content] = id;
            }
        for (const auto &name : {"<|im_start|>", "<|im_end|>", "<|im_sep|>"})
        {
            if (std::string(name) == "<|im_sep|>")
                continue;
            if (!special_.count(name))
                throw std::invalid_argument(std::string("Missing required special token: ") + name);
        }
    }
    std::size_t special(const std::string &name) const
    {
        auto it = special_.find(name);
        if (it == special_.end())
            throw std::out_of_range("Unknown special token: " + name);
        return it->second;
    }
    std::vector<std::size_t> encode(const std::string &text) const
    {
        std::vector<std::size_t> output;
        // Special tokens are intentionally treated as normal user text here.
        for (const auto &chunk : split(text))
        {
            auto ids = encodePiece(chunk);
            output.insert(output.end(), ids.begin(), ids.end());
        }
        return output;
    }
    std::string decode(const std::vector<std::size_t> &ids, bool skipSpecial = true) const
    {
        std::string out;
        for (auto id : ids)
        {
            if (id >= idToToken_.size() || idToToken_[id].empty())
            {
                throw std::out_of_range(
                    "Invalid token ID in decode: " + std::to_string(id));
            }
            const auto &token = idToToken_[id];
            if (skipOnDecode_.count(id))
            {
                if (!skipSpecial)
                    out += token;
                continue;
            }
            for (const auto &c : units(token))
            {
                auto it = symbolToByte_.find(c.cp);
                if (it == symbolToByte_.end())
                    throw std::invalid_argument("Invalid ByteLevel vocabulary symbol");
                out += static_cast<char>(it->second);
            }
        }
        return out;
    }
    // Serialize explicit user/assistant turns. The last turn must be a user
    // message; the assistant prefix is appended ready for generation.
    std::vector<std::size_t> chatConversation(
        const std::vector<std::pair<std::string, std::string>> &messages,
        bool thinking = false) const
    {
        if (messages.empty() || messages.back().first != "user")
            throw std::invalid_argument("Conversation must end with a user message.");
        std::vector<std::size_t> ids;
        auto add = [&](const std::string &name) { ids.push_back(special(name)); };
        auto plain = [&](const std::string &value) {
            const auto part = encode(value);
            ids.insert(ids.end(), part.begin(), part.end());
        };
        for (std::size_t i = 0; i < messages.size(); ++i)
        {
            const auto &message = messages[i];
            const auto &role = message.first;
            if (role != (i % 2 == 0 ? "user" : "assistant"))
                throw std::invalid_argument("Conversation roles must alternate user/assistant.");
            add("<|im_start|>");
            plain(role + "\n");
            plain(message.second);
            add("<|im_end|>");
            plain("\n");
        }
        add("<|im_start|>");
        plain("assistant\n");
        if (!thinking)
        {
            add("<think>");
            plain("\n\n");
            add("</think>");
            plain("\n\n");
        }
        return ids;
    }

    std::vector<std::size_t> chatPrompt(const std::string &userText, bool thinking = false) const
    {
        std::vector<std::size_t> ids;
        auto add = [&](const std::string &name)
        { ids.push_back(special(name)); };
        auto plain = [&](const std::string &text)
        {auto t=encode(text);ids.insert(ids.end(),t.begin(),t.end()); };
        add("<|im_start|>");
        plain("user\n");
        plain(userText);
        add("<|im_end|>");
        plain("\n");
        add("<|im_start|>");
        plain("assistant\n");
        // Qwen3 hard switch for enable_thinking=false: prefill an empty
        // reasoning block in the assistant turn, not /no_think in user text.
        if (!thinking)
        {
            add("<think>");
            plain("\n\n");
            add("</think>");
            plain("\n\n");
        }
        return ids;
    }
};
