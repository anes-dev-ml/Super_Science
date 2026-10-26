#pragma once

// Safetensors header validation, offsets and F32/F16/BF16 decoding.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <json.hpp>

struct SafeTensorInfo
{
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;

    std::uint64_t beginOffset;
    std::uint64_t endOffset;
};

inline std::size_t getSafeTensorElementSize(const std::string &dtype)
{
    if (dtype == "F32")
    {
        return 4;
    }
    if (dtype == "F16" || dtype == "BF16")
    {
        return 2;
    }
    throw std::invalid_argument("Unsupported Safetensors dtype: " + dtype);
}

inline float decodeBF16(std::uint16_t bits)
{
    const std::uint32_t floatBits = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &floatBits, sizeof(value));
    return value;
}

inline void validateSafeTensorStorage(const SafeTensorInfo &info)
{
    // The current Tensor class deliberately supports only positive dimensions.
    if (info.shape.empty() ||
        std::find(info.shape.begin(), info.shape.end(), std::size_t{0}) != info.shape.end())
    {
        throw std::invalid_argument("Scalar or empty Safetensors tensors are not supported by Tensor.");
    }
    const std::size_t elementSize = getSafeTensorElementSize(info.dtype);
    std::size_t elementCount = 1;
    for (const std::size_t dimension : info.shape)
    {
        if (elementCount >
            std::numeric_limits<std::size_t>::max() / dimension)
        {
            throw std::length_error("Tensor element count overflow.");
        }
        elementCount *= dimension;
    }

    if (elementCount != 0 &&
        elementSize > std::numeric_limits<std::size_t>::max() / elementCount)
    {
        throw std::length_error("Tensor byte count overflow.");
    }
    const std::size_t expectedBytes = elementCount * elementSize;

    if (info.beginOffset > info.endOffset)
    {
        throw std::invalid_argument("Invalid tensor offsets.");
    }
    const std::uint64_t actualBytes = info.endOffset - info.beginOffset;
    if (actualBytes > std::numeric_limits<std::size_t>::max() ||
        static_cast<std::size_t>(actualBytes) != expectedBytes)
    {
        throw std::invalid_argument("Tensor byte range does not match its shape and dtype.");
    }
}

inline void validateSafeTensorRanges(
    const std::vector<SafeTensorInfo> &tensors,
    std::uint64_t dataSectionSize)
{
    std::vector<SafeTensorInfo> sortedTensors = tensors;
    for (const SafeTensorInfo &tensor : sortedTensors)
    {
        validateSafeTensorStorage(tensor);
        if (tensor.endOffset > dataSectionSize)
        {
            throw std::invalid_argument("Tensor data range exceeds the data section.");
        }
    }

    std::sort(sortedTensors.begin(), sortedTensors.end(),
              [](const SafeTensorInfo &left, const SafeTensorInfo &right)
              {
                  if (left.beginOffset != right.beginOffset)
                  {
                      return left.beginOffset < right.beginOffset;
                  }
                  return left.endOffset < right.endOffset;
              });

    std::uint64_t coveredUntil = 0;
    for (const SafeTensorInfo &tensor : sortedTensors)
    {
        // Empty tensors consume no bytes and do not alter range coverage.
        if (tensor.beginOffset == tensor.endOffset)
        {
            continue;
        }
        if (tensor.beginOffset < coveredUntil)
        {
            throw std::invalid_argument("Safetensors data ranges overlap.");
        }
        if (tensor.beginOffset > coveredUntil)
        {
            throw std::invalid_argument("Gap in Safetensors data section.");
        }
        coveredUntil = tensor.endOffset;
    }

    if (coveredUntil != dataSectionSize)
    {
        throw std::invalid_argument("Tensor ranges do not cover the data section.");
    }
}

inline float decodeF16(std::uint16_t bits)
{
    const std::uint16_t sign = (bits >> 15) & 0x0001;
    const std::uint16_t exponent = (bits >> 10) & 0x001F;
    const std::uint16_t fraction = bits & 0x03FF;

    if (exponent == 0x001F)
    {
        if (fraction != 0)
        {
            return std::numeric_limits<float>::quiet_NaN();
        }
        const float infinity = std::numeric_limits<float>::infinity();
        return sign ? -infinity : infinity;
    }

    float value;
    if (exponent == 0)
    {
        // Subnormals include zero and scale the fraction by 2^-24.
        value = std::ldexp(static_cast<float>(fraction), -24);
    }
    else
    {
        const float significand = 1.0f + static_cast<float>(fraction) / 1024.0f;
        value = std::ldexp(significand, static_cast<int>(exponent) - 15);
    }

    return sign ? -value : value;
}

inline Tensor loadSafeTensor(
    const std::string& filePath,
    const SafeTensorInfo& info,
    std::uint64_t dataSectionStart
) {
    validateSafeTensorStorage(info);

    std::ifstream file(filePath, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Unable to open Safetensors file: " + filePath);
    }

    file.seekg(0, std::ios::end);
    const std::streampos fileEnd = file.tellg();
    if (fileEnd == std::streampos(-1))
    {
        throw std::runtime_error("Unable to determine Safetensors file size.");
    }
    const std::streamoff fileSize = static_cast<std::streamoff>(fileEnd);
    if (fileSize < 0)
    {
        throw std::runtime_error("Invalid Safetensors file size.");
    }

    if (info.endOffset > std::numeric_limits<std::uint64_t>::max() - dataSectionStart)
    {
        throw std::length_error("Safetensors file offset overflow.");
    }
    const std::uint64_t absoluteBegin = dataSectionStart + info.beginOffset;
    const std::uint64_t absoluteEnd = dataSectionStart + info.endOffset;
    if (absoluteEnd > static_cast<std::uint64_t>(fileSize))
    {
        throw std::invalid_argument("Tensor data range exceeds the file size.");
    }

    const std::uint64_t byteCount64 = info.endOffset - info.beginOffset;
    if (absoluteBegin > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        byteCount64 > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
        byteCount64 > std::numeric_limits<std::size_t>::max())
    {
        throw std::length_error("Tensor data range cannot be represented by the file stream.");
    }

    const std::size_t byteCount = static_cast<std::size_t>(byteCount64);
    std::vector<std::uint8_t> rawBytes(byteCount);
    file.seekg(static_cast<std::streamoff>(absoluteBegin), std::ios::beg);
    if (!file)
    {
        throw std::runtime_error("Unable to seek to tensor data.");
    }
    if (byteCount != 0)
    {
        const std::streamsize readSize = static_cast<std::streamsize>(byteCount);
        file.read(reinterpret_cast<char*>(rawBytes.data()), readSize);
        if (!file || file.gcount() != readSize)
        {
            throw std::runtime_error("Unable to read complete tensor data.");
        }
    }

    Tensor result(info.shape);
    const std::size_t elementSize = getSafeTensorElementSize(info.dtype);
    const std::size_t elementCount = byteCount / elementSize;
    float *output = result.dataPtr();
    for (std::size_t element = 0; element < elementCount; ++element)
    {
        const std::size_t offset = element * elementSize;
        float value;
        if (info.dtype == "F32")
        {
            const std::uint32_t bits =
                static_cast<std::uint32_t>(rawBytes[offset]) |
                (static_cast<std::uint32_t>(rawBytes[offset + 1]) << 8) |
                (static_cast<std::uint32_t>(rawBytes[offset + 2]) << 16) |
                (static_cast<std::uint32_t>(rawBytes[offset + 3]) << 24);
            std::memcpy(&value, &bits, sizeof(value));
        }
        else
        {
            const std::uint16_t bits =
                static_cast<std::uint16_t>(rawBytes[offset]) |
                (static_cast<std::uint16_t>(rawBytes[offset + 1]) << 8);
            value = info.dtype == "BF16" ? decodeBF16(bits) : decodeF16(bits);
        }

        output[element] = value;
    }
    return result;
}

inline std::vector<SafeTensorInfo> readSafeTensorHeader(
    const std::string& filePath
)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open Safetensors file: " + filePath);
    }

    const std::streamoff fileSize = file.tellg();
    if (fileSize < 8)
    {
        throw std::runtime_error("Safetensors file is too small to contain a header length.");
    }
    file.seekg(0, std::ios::beg);
    if (!file)
    {
        throw std::runtime_error("Failed to seek to the beginning of the Safetensors file.");
    }

    std::uint8_t bytes[8]{};
    file.read(reinterpret_cast<char*>(bytes), 8);
    if (!file)
    {
        throw std::runtime_error("Failed to read Safetensors header length.");
    }

    std::uint64_t headerLength = 0;
    for (std::size_t i = 0; i < 8; ++i)
    {
        headerLength |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }

    constexpr std::uint64_t MAX_HEADER_SIZE = 100ULL * 1024 * 1024;
    const std::uint64_t remainingFileSize =
        static_cast<std::uint64_t>(fileSize - 8);
    if (headerLength == 0 || headerLength > remainingFileSize ||
        headerLength > MAX_HEADER_SIZE)
    {
        throw std::runtime_error("Invalid Safetensors header length.");
    }

    std::string headerText(static_cast<std::size_t>(headerLength), '\0');
    file.read(headerText.data(), static_cast<std::streamsize>(headerLength));
    if (!file)
    {
        throw std::runtime_error("Failed to read Safetensors header.");
    }

    if (headerText.empty() || headerText.front() != '{')
    {
        throw std::invalid_argument(
            "Safetensors JSON header must begin with '{'.");
    }

    const nlohmann::json header = nlohmann::json::parse(headerText);
    if (!header.is_object())
    {
        throw std::runtime_error("Safetensors header must be a JSON object.");
    }

    const std::uint64_t dataSectionSize = remainingFileSize - headerLength;
    const auto readNonnegativeInteger = [](const nlohmann::json& value,
                                           const std::string& field) -> std::uint64_t
    {
        if (value.is_number_unsigned())
        {
            return value.get<std::uint64_t>();
        }
        if (!value.is_number_integer())
        {
            throw std::invalid_argument(field + " must be a nonnegative integer.");
        }
        const std::int64_t signedValue = value.get<std::int64_t>();
        if (signedValue < 0)
        {
            throw std::invalid_argument(field + " must be a nonnegative integer.");
        }
        return static_cast<std::uint64_t>(signedValue);
    };

    std::vector<SafeTensorInfo> tensors;
    for (auto entry = header.begin(); entry != header.end(); ++entry)
    {
        if (entry.key() == "__metadata__")
        {
            continue;
        }
        const auto& tensor = entry.value();
        if (!tensor.is_object() || !tensor.contains("dtype") ||
            !tensor.contains("shape") || !tensor.contains("data_offsets"))
        {
            throw std::invalid_argument("Malformed Safetensors entry: " + entry.key());
        }
        if (!tensor.at("dtype").is_string() || !tensor.at("shape").is_array() ||
            !tensor.at("data_offsets").is_array() || tensor.at("data_offsets").size() != 2)
        {
            throw std::invalid_argument("Malformed Safetensors entry: " + entry.key());
        }

        SafeTensorInfo info{};
        info.name = entry.key();
        info.dtype = tensor.at("dtype").get<std::string>();
        for (const auto& dimension : tensor.at("shape"))
        {
            const std::uint64_t value = readNonnegativeInteger(dimension, "Tensor dimension");
            if (value > std::numeric_limits<std::size_t>::max())
            {
                throw std::out_of_range("Tensor dimension exceeds size_t range.");
            }
            info.shape.push_back(static_cast<std::size_t>(value));
        }

        const auto& offsets = tensor.at("data_offsets");
        info.beginOffset = readNonnegativeInteger(offsets[0], "Tensor begin offset");
        info.endOffset = readNonnegativeInteger(offsets[1], "Tensor end offset");
        if (info.beginOffset > info.endOffset || info.endOffset > dataSectionSize)
        {
            throw std::invalid_argument("Invalid Safetensors data offsets for: " + info.name);
        }
        tensors.push_back(std::move(info));
    }

    validateSafeTensorRanges(tensors, dataSectionSize);
    return tensors;
}
