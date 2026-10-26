#pragma once

// Qwen model configuration: structure, JSON decoding, and validation.
#include <json.hpp>
#include <fstream>
#include <cstdint>
#include <string>

struct ModelConfig
{
    std::size_t hiddenSize;
    std::size_t numLayers;
    std::size_t numQueryHeads;
    std::size_t numKVHeads;
    std::size_t headDim;
    std::size_t intermediateSize;
    std::size_t vocabSize;

    float rmsNormEpsilon;
    float ropeTheta;

    static ModelConfig loadModelConfig(const std::string &filePath);

};

inline ModelConfig qwen3_06b{
    1024,       // hiddenSize
    28,         // numLayers
    16,         // numQueryHeads
    8,          // numKVHeads
    128,        // headDim
    3072,       // intermediateSize
    151936,     // vocabSize
    1e-6f,      // rmsNormEpsilon
    1000000.0f  // ropeTheta
};

inline void validateModelConfig(const ModelConfig &config)
{
    if (config.hiddenSize == 0)
    {
        throw std::invalid_argument("hiddenSize must be positive.");
    }
    if (config.numLayers == 0)
    {
        throw std::invalid_argument("numLayers must be positive.");
    }
    if (config.numQueryHeads == 0)
    {
        throw std::invalid_argument("numQueryHeads must be positive.");
    }
    if (config.numKVHeads == 0)
    {
        throw std::invalid_argument("numKVHeads must be positive.");
    }
    if (config.numQueryHeads % config.numKVHeads != 0)
    {
        throw std::invalid_argument("numQueryHeads must be divisible by numKVHeads.");
    }
    if (config.headDim == 0)
    {
        throw std::invalid_argument("headDim must be positive.");
    }
    if (config.headDim % 2 != 0)
    {
        throw std::invalid_argument("headDim must be even for RoPE.");
    }
    if (config.intermediateSize == 0)
    {
        throw std::invalid_argument("intermediateSize must be positive.");
    }
    if (config.vocabSize == 0)
    {
        throw std::invalid_argument("vocabSize must be positive.");
    }
    if (!std::isfinite(config.rmsNormEpsilon) || config.rmsNormEpsilon <= 0.0f)
    {
        throw std::invalid_argument("rmsNormEpsilon must be positive and finite.");
    }
    if (!std::isfinite(config.ropeTheta) || config.ropeTheta <= 0.0f)
    {
        throw std::invalid_argument("ropeTheta must be positive and finite.");
    }

    const std::size_t maxDimension = std::numeric_limits<std::size_t>::max();
    if (config.numQueryHeads > maxDimension / config.headDim)
    {
        throw std::length_error("Query projection dimension overflow.");
    }
    if (config.numKVHeads > maxDimension / config.headDim)
    {
        throw std::length_error("KV projection dimension overflow.");
    }
}


inline std::size_t readPositiveSize(
    const nlohmann::json& document,
    const std::string& key
)
{
    const auto& value = document.at(key);

    if (!value.is_number_unsigned() &&
        !value.is_number_integer())
    {
        throw std::invalid_argument(
            key + " must be an integer."
        );
    }

    if (value.is_number_integer() &&
        value.get<std::int64_t>() <= 0)
    {
        throw std::invalid_argument(
            key + " must be positive."
        );
    }

    const std::uint64_t number =
        value.get<std::uint64_t>();

    if (number > std::numeric_limits<std::size_t>::max())
    {
        throw std::out_of_range(
            key + " exceeds size_t range."
        );
    }

    return static_cast<std::size_t>(number);
}

inline ModelConfig loadModelConfig(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open model configuration file: " + filePath);
    }

    nlohmann::json document;
    file >> document;
    if (!document.is_object())
    {
        throw std::runtime_error("Model configuration must be a JSON object.");
    }
    if (document.at("model_type").get<std::string>() != "qwen3")
    {
        throw std::runtime_error("Unsupported model_type; expected qwen3.");
    }

    ModelConfig config{
        readPositiveSize(document, "hidden_size"),
        readPositiveSize(document, "num_hidden_layers"),
        readPositiveSize(document, "num_attention_heads"),
        readPositiveSize(document, "num_key_value_heads"),
        readPositiveSize(document, "head_dim"),
        readPositiveSize(document, "intermediate_size"),
        readPositiveSize(document, "vocab_size"),
        document.at("rms_norm_eps").get<float>(),
        document.at("rope_theta").get<float>()};

    validateModelConfig(config);
    return config;
}

// Keep the original class API callable as well as the existing free function.
inline ModelConfig ModelConfig::loadModelConfig(const std::string &filePath)
{
    return ::loadModelConfig(filePath);
}
