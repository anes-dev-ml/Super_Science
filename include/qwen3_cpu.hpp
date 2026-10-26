#pragma once

// A deliberately simple, float32 CPU reference path for dense Qwen3.
// It accepts token IDs; tokenization is kept separate from model execution.
#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Qwen3LayerWeights
{
    Tensor inputNorm, postAttentionNorm;
    Tensor queryNorm, keyNorm;
    Tensor query, key, value, output;
    Tensor gate, up, down;
};

class Qwen3Cpu
{
public:
    Qwen3Cpu(ModelConfig config, Tensor embedding,
             std::vector<Qwen3LayerWeights> layers, Tensor finalNorm)
        : config_(config), embedding_(std::move(embedding)),
          layers_(std::move(layers)), finalNorm_(std::move(finalNorm)),
          caches_(config.numLayers)
    {
        validateModelConfig(config_);
        inverseFrequency_.resize(config_.headDim / 2);
        for (std::size_t d = 0; d < inverseFrequency_.size(); ++d)
        {
            inverseFrequency_[d] = 1.0 /
                std::pow(static_cast<double>(config_.ropeTheta),
                         2.0 * static_cast<double>(d) / config_.headDim);
        }
        expectShape(embedding_, {config_.vocabSize, config_.hiddenSize}, "embed_tokens");
        expectShape(finalNorm_, {config_.hiddenSize}, "final norm");
        if (layers_.size() != config_.numLayers)
            throw std::invalid_argument("Layer count does not match the configuration.");

        const std::size_t qWidth = config_.numQueryHeads * config_.headDim;
        const std::size_t kvWidth = config_.numKVHeads * config_.headDim;
        for (const auto &layer : layers_)
        {
            expectShape(layer.inputNorm, {config_.hiddenSize}, "input norm");
            expectShape(layer.postAttentionNorm, {config_.hiddenSize}, "post-attention norm");
            expectShape(layer.queryNorm, {config_.headDim}, "query norm");
            expectShape(layer.keyNorm, {config_.headDim}, "key norm");
            expectShape(layer.query, {qWidth, config_.hiddenSize}, "query projection");
            expectShape(layer.key, {kvWidth, config_.hiddenSize}, "key projection");
            expectShape(layer.value, {kvWidth, config_.hiddenSize}, "value projection");
            expectShape(layer.output, {config_.hiddenSize, qWidth}, "output projection");
            expectShape(layer.gate, {config_.intermediateSize, config_.hiddenSize}, "gate projection");
            expectShape(layer.up, {config_.intermediateSize, config_.hiddenSize}, "up projection");
            expectShape(layer.down, {config_.hiddenSize, config_.intermediateSize}, "down projection");
        }
    }

    static Qwen3Cpu fromDirectory(const std::filesystem::path &directory)
    {
        const ModelConfig config = loadModelConfig((directory / "config.json").string());
        const std::string weightsFile = (directory / "model.safetensors").string();
        const auto metadata = readSafeTensorHeader(weightsFile);
        std::unordered_map<std::string, SafeTensorInfo> index;
        for (const auto &entry : metadata)
            if (!index.emplace(entry.name, entry).second)
                throw std::invalid_argument("Duplicate Safetensors tensor name: " + entry.name);

        // Header is 8-byte little-endian size followed by JSON metadata.
        // This is the start of the data section, not the start of the file.
        std::ifstream input(weightsFile, std::ios::binary);
        if (!input)
            throw std::runtime_error("Cannot open model weights.");
        std::uint64_t jsonLength = 0;
        for (unsigned shift = 0; shift < 64; shift += 8)
        {
            int byte = input.get();
            if (byte == EOF)
                throw std::runtime_error("Truncated Safetensors file.");
            jsonLength |= static_cast<std::uint64_t>(byte) << shift;
        }
        if (jsonLength > std::numeric_limits<std::uint64_t>::max() - 8)
            throw std::length_error("Safetensors header offset overflow.");
        const std::uint64_t dataStart = jsonLength + 8;

        auto fetch = [&](const std::string &name, std::vector<std::size_t> shape)
        {
            const auto it = index.find(name);
            if (it == index.end())
                throw std::invalid_argument("Missing model weight: " + name);
            if (it->second.shape != shape)
                throw std::invalid_argument("Incorrect model weight shape: " + name);
            return loadSafeTensor(weightsFile, it->second, dataStart);
        };

        Tensor embedding = fetch("model.embed_tokens.weight", {config.vocabSize, config.hiddenSize});
        std::vector<Qwen3LayerWeights> layers;
        layers.reserve(config.numLayers);
        const std::size_t queryWidth = config.numQueryHeads * config.headDim;
        const std::size_t kvWidth = config.numKVHeads * config.headDim;
        for (std::size_t i = 0; i < config.numLayers; ++i)
        {
            const std::string prefix = "model.layers." + std::to_string(i) + ".";
            const std::string attention = prefix + "self_attn.";
            const std::string mlp = prefix + "mlp.";
            layers.push_back({
                fetch(prefix + "input_layernorm.weight", {config.hiddenSize}),
                fetch(prefix + "post_attention_layernorm.weight", {config.hiddenSize}),
                fetch(attention + "q_norm.weight", {config.headDim}),
                fetch(attention + "k_norm.weight", {config.headDim}),
                fetch(attention + "q_proj.weight", {queryWidth, config.hiddenSize}),
                fetch(attention + "k_proj.weight", {kvWidth, config.hiddenSize}),
                fetch(attention + "v_proj.weight", {kvWidth, config.hiddenSize}),
                fetch(attention + "o_proj.weight", {config.hiddenSize, queryWidth}),
                fetch(mlp + "gate_proj.weight", {config.intermediateSize, config.hiddenSize}),
                fetch(mlp + "up_proj.weight", {config.intermediateSize, config.hiddenSize}),
                fetch(mlp + "down_proj.weight", {config.hiddenSize, config.intermediateSize})});
        }
        Tensor finalNorm = fetch("model.norm.weight", {config.hiddenSize});
        return Qwen3Cpu(config, std::move(embedding), std::move(layers), std::move(finalNorm));
    }

    void reset()
    {
        position_ = 0;
        for (auto &cache : caches_)
        {
            cache.keys.clear();
            cache.values.clear();
        }
    }

    std::size_t position() const noexcept { return position_; }

    // Processes exactly one token and returns its next-token logits.
    std::vector<float> forwardToken(std::size_t token)
    {
        return step(token, true);
    }

    std::vector<std::size_t> generate(const std::vector<std::size_t> &prompt,
                                      std::size_t newTokens)
    {
        if (prompt.empty())
            throw std::invalid_argument("Prompt must have at least one token.");
        reset();
        std::vector<std::size_t> result = prompt;
        std::vector<float> logits;
        for (std::size_t i = 0; i < prompt.size(); ++i)
            logits = step(prompt[i], i + 1 == prompt.size() && newTokens != 0);
        for (std::size_t n = 0; n < newTokens; ++n)
        {
            const auto next = static_cast<std::size_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            result.push_back(next);
            if (next == 151643 || next == 151645) // Qwen3 EOS markers
                break;
            if (n + 1 < newTokens)
                logits = step(next, true);
        }
        return result;
    }

private:
    struct Cache { std::vector<float> keys, values; };
    ModelConfig config_;
    Tensor embedding_;
    std::vector<Qwen3LayerWeights> layers_;
    Tensor finalNorm_;
    std::vector<Cache> caches_;
    std::vector<double> inverseFrequency_;
    std::size_t position_ = 0;

    // The tied language-model head is expensive. Skip it for non-final prompt
    // tokens where only the KV-cache and hidden states are needed.
    std::vector<float> step(std::size_t token, bool returnLogits)
    {
        if (token >= config_.vocabSize)
            throw std::out_of_range("Token ID exceeds vocabulary size.");
        if (position_ == std::numeric_limits<std::size_t>::max())
            throw std::length_error("Position counter overflow.");
        const std::size_t dim = config_.hiddenSize;
        const float *tokenRow = embedding_.dataPtr() + token * dim;
        std::vector<float> hidden(tokenRow, tokenRow + dim);

        for (std::size_t i = 0; i < layers_.size(); ++i)
        {
            const auto &w = layers_[i];
            auto &cache = caches_[i];
            const auto normed = rmsNorm(hidden, w.inputNorm);
            auto q = matvec(w.query, normed);
            auto k = matvec(w.key, normed);
            auto v = matvec(w.value, normed);
            normalizeHeads(q, config_.numQueryHeads, w.queryNorm);
            normalizeHeads(k, config_.numKVHeads, w.keyNorm);
            rotateHeads(q, config_.numQueryHeads, position_);
            rotateHeads(k, config_.numKVHeads, position_);

            cache.keys.insert(cache.keys.end(), k.begin(), k.end());
            cache.values.insert(cache.values.end(), v.begin(), v.end());

            std::vector<float> attended(config_.numQueryHeads * config_.headDim, 0.0f);
            const std::size_t groupSize = config_.numQueryHeads / config_.numKVHeads;
            const std::size_t sequenceLength = position_ + 1;
            const std::size_t kvStride = config_.numKVHeads * config_.headDim;
            const float scale = 1.0f / std::sqrt(static_cast<float>(config_.headDim));
            for (std::size_t h = 0; h < config_.numQueryHeads; ++h)
            {
                const std::size_t kv = h / groupSize;
                const float *query = q.data() + h * config_.headDim;
                std::vector<float> scores(sequenceLength);
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t t = 0; t < sequenceLength; ++t)
                {
                    const float *key = cache.keys.data() + t * kvStride + kv * config_.headDim;
                    float dot = 0.0f;
                    for (std::size_t d = 0; d < config_.headDim; ++d)
                        dot += query[d] * key[d];
                    scores[t] = dot * scale;
                    maximum = std::max(maximum, scores[t]);
                }
                float denominator = 0.0f;
                for (auto &score : scores)
                {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                if (!std::isfinite(denominator) || denominator <= 0.0f)
                    throw std::runtime_error("Invalid attention softmax denominator.");
                float *headOutput = attended.data() + h * config_.headDim;
                for (std::size_t t = 0; t < sequenceLength; ++t)
                {
                    const float *value = cache.values.data() + t * kvStride + kv * config_.headDim;
                    const float probability = scores[t] / denominator;
                    for (std::size_t d = 0; d < config_.headDim; ++d)
                        headOutput[d] += probability * value[d];
                }
            }

            const auto attentionOutput = matvec(w.output, attended);
            for (std::size_t d = 0; d < dim; ++d)
                hidden[d] += attentionOutput[d];

            const auto feedforwardInput = rmsNorm(hidden, w.postAttentionNorm);
            auto gate = matvec(w.gate, feedforwardInput);
            const auto up = matvec(w.up, feedforwardInput);
            for (std::size_t d = 0; d < gate.size(); ++d)
                gate[d] = (gate[d] / (1.0f + std::exp(-gate[d]))) * up[d];
            const auto down = matvec(w.down, gate);
            for (std::size_t d = 0; d < dim; ++d)
                hidden[d] += down[d];
        }
        ++position_;
        if (!returnLogits)
            return {};
        return matvec(embedding_, rmsNorm(hidden, finalNorm_)); // tied LM head
    }

    static void expectShape(const Tensor &tensor, std::vector<std::size_t> shape,
                            const char *name)
    {
        if (tensor.getShape() != shape)
            throw std::invalid_argument(std::string("Incorrect tensor shape: ") + name);
    }

    static std::vector<float> matvec(const Tensor &matrix,
                                     const std::vector<float> &input)
    {
        const auto &shape = matrix.getShape();
        if (shape.size() != 2 || shape[1] != input.size())
            throw std::invalid_argument("matvec dimension mismatch.");
        std::vector<float> output(shape[0]);
        const float *weight = matrix.dataPtr();
        for (std::size_t row = 0; row < shape[0]; ++row)
        {
            const float *values = weight + row * shape[1];
            float dot = 0.0f;
            for (std::size_t col = 0; col < shape[1]; ++col)
                dot += values[col] * input[col];
            output[row] = dot;
        }
        return output;
    }

    std::vector<float> rmsNorm(const std::vector<float> &input,
                                const Tensor &weight) const
    {
        if (weight.getSize() != input.size())
            throw std::invalid_argument("RMSNorm width mismatch.");
        float squares = 0.0f;
        for (float x : input)
            squares += x * x;
        const float factor = 1.0f / std::sqrt(
            squares / static_cast<float>(input.size()) + config_.rmsNormEpsilon);
        std::vector<float> output(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
            output[i] = input[i] * factor * weight.dataPtr()[i];
        return output;
    }

    void normalizeHeads(std::vector<float> &values, std::size_t heads,
                        const Tensor &weight) const
    {
        for (std::size_t head = 0; head < heads; ++head)
        {
            const std::size_t start = head * config_.headDim;
            float squares = 0.0f;
            for (std::size_t d = 0; d < config_.headDim; ++d)
                squares += values[start + d] * values[start + d];
            const float factor = 1.0f / std::sqrt(
                squares / static_cast<float>(config_.headDim) + config_.rmsNormEpsilon);
            for (std::size_t d = 0; d < config_.headDim; ++d)
                values[start + d] *= factor * weight.dataPtr()[d];
        }
    }

    // Qwen/HF rotate_half: pair the first and second halves, not adjacent elements.
    void rotateHeads(std::vector<float> &values, std::size_t heads,
                     std::size_t position) const
    {
        const std::size_t half = config_.headDim / 2;
        for (std::size_t head = 0; head < heads; ++head)
        {
            const std::size_t start = head * config_.headDim;
            for (std::size_t d = 0; d < half; ++d)
            {
                const double theta = static_cast<double>(position) * inverseFrequency_[d];
                const float cosine = static_cast<float>(std::cos(theta));
                const float sine = static_cast<float>(std::sin(theta));
                const float a = values[start + d];
                const float b = values[start + half + d];
                values[start + d] = a * cosine - b * sine;
                values[start + half + d] = b * cosine + a * sine;
            }
        }
    }
};
