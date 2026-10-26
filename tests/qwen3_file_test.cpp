#include "qwen3_cpu.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using NamedTensor = std::pair<std::string, Tensor>;

void writeLE(std::ostream &out, std::uint64_t bits, unsigned bytes)
{
    for (unsigned i = 0; i < bytes; ++i)
        out.put(static_cast<char>((bits >> (8 * i)) & 0xff));
}

struct TemporaryDirectory
{
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ss_qwen3_fixture_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void writeModel(const std::filesystem::path &directory, bool missingWeight)
{
    nlohmann::json config = {
        {"model_type", "qwen3"}, {"hidden_size", 4},
        {"num_hidden_layers", 1}, {"num_attention_heads", 2},
        {"num_key_value_heads", 1}, {"head_dim", 2},
        {"intermediate_size", 8}, {"vocab_size", 4},
        {"rms_norm_eps", 1e-6}, {"rope_theta", 1000000.0}};
    std::ofstream(directory / "config.json") << config.dump();

    Tensor embedding({4, 4});
    for (std::size_t i = 0; i < 4; ++i)
        embedding.at({i, i}) = 1.0f;
    Tensor norm4({4});
    Tensor norm2({2});
    for (std::size_t i = 0; i < 4; ++i)
        norm4.at({i}) = 1.0f;
    for (std::size_t i = 0; i < 2; ++i)
        norm2.at({i}) = 1.0f;

    const std::string base = "model.layers.0.";
    const std::string attn = base + "self_attn.";
    const std::string mlp = base + "mlp.";
    std::vector<NamedTensor> tensors{
        {"model.embed_tokens.weight", embedding},
        {base + "input_layernorm.weight", norm4},
        {base + "post_attention_layernorm.weight", norm4},
        {attn + "q_norm.weight", norm2},
        {attn + "k_norm.weight", norm2},
        {attn + "q_proj.weight", Tensor({4, 4})},
        {attn + "k_proj.weight", Tensor({2, 4})},
        {attn + "v_proj.weight", Tensor({2, 4})},
        {attn + "o_proj.weight", Tensor({4, 4})},
        {mlp + "gate_proj.weight", Tensor({8, 4})},
        {mlp + "up_proj.weight", Tensor({8, 4})},
        {mlp + "down_proj.weight", Tensor({4, 8})},
        {"model.norm.weight", norm4}};
    if (missingWeight)
        tensors.pop_back();

    nlohmann::json header = nlohmann::json::object();
    std::size_t cursor = 0;
    for (const auto &entry : tensors)
    {
        const std::size_t bytes = entry.second.getSize() * 4;
        header[entry.first] = {
            {"dtype", "F32"}, {"shape", entry.second.getShape()},
            {"data_offsets", {cursor, cursor + bytes}}};
        cursor += bytes;
    }
    const std::string text = header.dump();
    std::ofstream file(directory / "model.safetensors", std::ios::binary);
    writeLE(file, text.size(), 8);
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto &entry : tensors)
    {
        for (std::size_t i = 0; i < entry.second.getSize(); ++i)
        {
            std::uint32_t bits;
            const float value = entry.second.dataPtr()[i];
            std::memcpy(&bits, &value, sizeof(bits));
            writeLE(file, bits, 4);
        }
    }
    if (!file)
        throw std::runtime_error("Cannot write synthetic model fixture.");
}
}

int main()
{
    try
    {
        TemporaryDirectory temporary;
        writeModel(temporary.path, false);
        Qwen3Cpu model = Qwen3Cpu::fromDirectory(temporary.path);
        const auto result = model.generate({3, 1}, 2);
        if (result != std::vector<std::size_t>{3, 1, 1, 1})
            throw std::runtime_error("Incorrect logits from loaded synthetic weights.");

        writeModel(temporary.path, true);
        bool rejected = false;
        try
        {
            Qwen3Cpu::fromDirectory(temporary.path);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Missing layer weight was not rejected.");
        std::cout << "Qwen3 model-file loading and missing-weight rejection PASSED!\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
