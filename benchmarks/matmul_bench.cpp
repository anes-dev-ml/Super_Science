// Reproducible comparison of educational Tensor matrix multiplication variants.
// Not an inference latency benchmark; run in Release mode on a fixed CPU.
#include "tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
std::size_t dimension(const char *text)
{
    const std::string input(text);
    std::size_t parsed = 0;
    const auto value = std::stoull(input, &parsed);
    if (parsed != input.size() || value == 0 || value > 512)
        throw std::invalid_argument("Dimensions must be integers from 1 to 512.");
    return static_cast<std::size_t>(value);
}

void verifyClose(const Tensor &reference, const Tensor &actual,
                 const std::string &name)
{
    if (actual.getShape() != reference.getShape())
        throw std::runtime_error(name + " returned an incorrect shape.");
    for (std::size_t i = 0; i < reference.getSize(); ++i)
    {
        const float expected = reference.dataPtr()[i];
        const float found = actual.dataPtr()[i];
        const float tolerance = 1e-3f + 1e-4f * std::fabs(expected);
        if (!std::isfinite(found) || std::fabs(found - expected) > tolerance)
            throw std::runtime_error(name + " disagrees with reference at offset " +
                                     std::to_string(i));
    }
}

template <class Operation>
double medianMilliseconds(Operation operation, volatile float &sink)
{
    using Clock = std::chrono::steady_clock;
    // One warmup run, followed by five complete independent measurements.
    sink = sink + operation().dataPtr()[0];
    std::vector<double> samples;
    for (int i = 0; i < 5; ++i)
    {
        const auto start = Clock::now();
        Tensor result = operation();
        const auto finish = Clock::now();
        sink = sink + result.dataPtr()[0]; // Prevent dead-code elimination.
        samples.push_back(
            std::chrono::duration<double, std::milli>(finish - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc != 1 && argc != 4)
            throw std::invalid_argument("Usage: matmul_bench [rows inner columns]");

        const std::size_t rows = argc == 4 ? dimension(argv[1]) : 128;
        const std::size_t inner = argc == 4 ? dimension(argv[2]) : 128;
        const std::size_t cols = argc == 4 ? dimension(argv[3]) : 128;

        Tensor a({rows, inner});
        Tensor b({inner, cols});
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t k = 0; k < inner; ++k)
                a.at({i, k}) = std::sin(static_cast<float>(i * 13 + k * 7) * 0.02f);
        for (std::size_t k = 0; k < inner; ++k)
            for (std::size_t j = 0; j < cols; ++j)
                b.at({k, j}) = std::cos(static_cast<float>(k * 5 + j * 11) * 0.02f);

        const Tensor expected = a.matmul(b);
        verifyClose(expected, a.matmulDirect(b), "matmulDirect");
        verifyClose(expected, a.matmulIKJ(b), "matmulIKJ");
        verifyClose(expected, a.matmulBlocked(b), "matmulBlocked");

        volatile float sink = 0.0f;
        std::cout << "Matrix dimensions: " << rows << " x " << inner
                  << " times " << inner << " x " << cols << '\n';
        std::cout << "All four implementations agree within tolerance.\n";
        std::cout << "Release build recommended; times are median of 5 runs (ms).\n";
        std::cout << std::fixed << std::setprecision(3);

        const auto report = [&](const char *name, auto operation)
        {
            std::cout << std::setw(15) << name << ": "
                      << medianMilliseconds(operation, sink) << " ms\n";
        };
        report("matmul", [&] { return a.matmul(b); });
        report("matmulDirect", [&] { return a.matmulDirect(b); });
        report("matmulIKJ", [&] { return a.matmulIKJ(b); });
        report("matmulBlocked", [&] { return a.matmulBlocked(b); });
        std::cout << "Verification sink: " << sink << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
