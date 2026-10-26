
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.hpp"

using json = nlohmann::json;

// Tests should never overwrite project files or leave generated weight files.
struct TemporarySafeTensorFile
{
    std::filesystem::path path;

    explicit TemporarySafeTensorFile(const std::string &dtype)
        : path(std::filesystem::temp_directory_path() /
               ("super_science_" + dtype + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                ".safetensors"))
    {
    }

    ~TemporarySafeTensorFile()
    {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

// Write an unsigned integer using little-endian byte order.
void writeLittleEndian(
    std::ofstream& file,
    std::uint64_t value,
    std::size_t byteCount)
{
    for (std::size_t i = 0; i < byteCount; ++i)
    {
        file.put(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

// Create a small, valid Safetensors file.
// The tensor has shape [2, 2] and values:
// [1, 2]
// [3, 4]
void createTestSafeTensorFile(
    const std::string& filePath,
    const std::string& dtype)
{
    std::size_t bytesPerElement = getSafeTensorElementSize(dtype);

    json header;
    header["test.weight"] = {
        {"dtype", dtype},
        {"shape", {2, 2}},
        {"data_offsets", {0, 4 * bytesPerElement}}
    };

    std::string headerText = header.dump();

    std::ofstream file(filePath, std::ios::binary | std::ios::trunc);

    if (!file)
    {
        throw std::runtime_error("Failed to create test file.");
    }

    // Safetensors header length: 8-byte little-endian uint64.
    writeLittleEndian(file, headerText.size(), 8);

    // JSON header.
    file.write(
        headerText.data(),
        static_cast<std::streamsize>(headerText.size())
    );

    if (dtype == "F32")
    {
        const std::uint32_t values[] = {
            0x3F800000, // 1.0
            0x40000000, // 2.0
            0x40400000, // 3.0
            0x40800000  // 4.0
        };

        for (std::uint32_t bits : values)
        {
            writeLittleEndian(file, bits, 4);
        }
    }
    else if (dtype == "BF16")
    {
        const std::uint16_t values[] = {
            0x3F80, // 1.0
            0x4000, // 2.0
            0x4040, // 3.0
            0x4080  // 4.0
        };

        for (std::uint16_t bits : values)
        {
            writeLittleEndian(file, bits, 2);
        }
    }
    else if (dtype == "F16")
    {
        const std::uint16_t values[] = {
            0x3C00, // 1.0
            0x4000, // 2.0
            0x4200, // 3.0
            0x4400  // 4.0
        };

        for (std::uint16_t bits : values)
        {
            writeLittleEndian(file, bits, 2);
        }
    }

    if (!file)
    {
        throw std::runtime_error("Failed to write test tensor data.");
    }
}

// Read the 8-byte header length to calculate where tensor data starts.
std::uint64_t getDataSectionStart(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Unable to open test file.");
    }

    std::uint64_t headerLength = 0;

    for (std::size_t i = 0; i < 8; ++i)
    {
        const int byte = file.get();

        if (byte == EOF)
        {
            throw std::runtime_error("Incomplete Safetensors header length.");
        }

        headerLength |=
            static_cast<std::uint64_t>(
                static_cast<unsigned char>(byte)
            ) << (8 * i);
    }

    if (headerLength > std::numeric_limits<std::uint64_t>::max() - 8)
    {
        throw std::length_error("Header offset overflow.");
    }

    return 8 + headerLength;
}

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error("TEST FAILED: " + message);
    }
}

// End-to-end test for a particular dtype.
void testSafeTensorLoading(const std::string& dtype)
{
    TemporarySafeTensorFile temporary(dtype);
    const std::string filePath = temporary.path.string();

    createTestSafeTensorFile(filePath, dtype);

    const auto metadata = readSafeTensorHeader(filePath);

    require(metadata.size() == 1, "Expected exactly one tensor.");
    require(metadata[0].name == "test.weight", "Wrong tensor name.");
    require(metadata[0].dtype == dtype, "Wrong dtype.");
    require(
        metadata[0].shape == std::vector<std::size_t>({2, 2}),
        "Wrong tensor shape."
    );

    const std::uint64_t dataSectionStart =
        getDataSectionStart(filePath);

    const std::uint64_t expectedBytes =
        4 * getSafeTensorElementSize(dtype);

    require(metadata[0].beginOffset == 0, "Wrong begin offset.");
    require(metadata[0].endOffset == expectedBytes, "Wrong end offset.");

    validateSafeTensorRanges(metadata, expectedBytes);

    Tensor loaded = loadSafeTensor(
        filePath,
        metadata[0],
        dataSectionStart
    );

    require(
        loaded.getShape() == std::vector<std::size_t>({2, 2}),
        "Loaded tensor has wrong shape."
    );

    const float expected[2][2] = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    for (std::size_t row = 0; row < 2; ++row)
    {
        for (std::size_t col = 0; col < 2; ++col)
        {
            const float actual = loaded.at({row, col});

            require(
                std::fabs(actual - expected[row][col]) < 1e-6f,
                "Incorrect decoded value for " + dtype
            );
        }
    }

    std::cout << dtype << " Safetensors loading PASSED!\n";
}

void testF16SpecialValues()
{
    require(decodeF16(0x3C00) == 1.0f, "F16 1.0");
    require(decodeF16(0x4000) == 2.0f, "F16 2.0");
    require(decodeF16(0xBC00) == -1.0f, "F16 -1.0");
    require(decodeF16(0x3800) == 0.5f, "F16 0.5");

    require(decodeF16(0x0000) == 0.0f, "F16 +0");
    require(!std::signbit(decodeF16(0x0000)), "F16 positive zero sign");

    require(decodeF16(0x8000) == 0.0f, "F16 -0");
    require(std::signbit(decodeF16(0x8000)), "F16 negative zero sign");

    require(std::isinf(decodeF16(0x7C00)), "F16 +inf");
    require(decodeF16(0x7C00) > 0, "F16 +inf sign");

    require(std::isinf(decodeF16(0xFC00)), "F16 -inf");
    require(decodeF16(0xFC00) < 0, "F16 -inf sign");

    require(std::isnan(decodeF16(0x7E00)), "F16 NaN");

    require(decodeF16(0x0001) == std::ldexp(1.0f, -24),
            "F16 smallest subnormal");

    require(decodeF16(0x7BFF) == 65504.0f,
            "F16 largest finite value");

    std::cout << "F16 special-value decoding PASSED!\n";
}

void testBF16SpecialValues()
{
    require(decodeBF16(0x3F80) == 1.0f, "BF16 1.0");
    require(decodeBF16(0x4000) == 2.0f, "BF16 2.0");
    require(decodeBF16(0xBF80) == -1.0f, "BF16 -1.0");
    require(decodeBF16(0x3F00) == 0.5f, "BF16 0.5");

    require(decodeBF16(0x0000) == 0.0f, "BF16 +0");
    require(std::signbit(decodeBF16(0x8000)), "BF16 negative zero");

    require(std::isinf(decodeBF16(0x7F80)), "BF16 +inf");
    require(std::isinf(decodeBF16(0xFF80)), "BF16 -inf");
    require(std::isnan(decodeBF16(0x7FC0)), "BF16 NaN");

    std::cout << "BF16 special-value decoding PASSED!\n";
}

int main()
{
    try
    {
        testF16SpecialValues();
        testBF16SpecialValues();

        testSafeTensorLoading("F32");
        testSafeTensorLoading("BF16");
        testSafeTensorLoading("F16");

        std::cout << "\nALL SAFETENSORS TESTS PASSED!\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "\n" << error.what() << '\n';
        return 1;
    }
}
