// Manual full-model HIP gate for static, single-float-input/output exports.
// Usage: hip_model_check model.onnx input.bin ort-reference.bin
// Build against the HIP-enabled nert archive with hipcc (see AMD runbook).
#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/model_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<float> Read(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error(std::string("cannot open ") + path);
    const auto bytes = file.tellg();
    if (bytes <= 0 || bytes % sizeof(float) != 0)
        throw std::runtime_error("invalid float32 file length");
    std::vector<float> data(static_cast<std::size_t>(bytes) / sizeof(float));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), bytes))
        throw std::runtime_error("short read");
    return data;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: hip_model_check model.onnx input.bin ort-reference.bin\n";
        return 2;
    }
    try {
        const auto graph = nert::LoadGraphFromFile(argv[1]);
        if (graph.inputs.size() != 1 || graph.outputs.size() != 1 ||
            graph.inputs[0].second.dtype != nert::DataType::Float32 ||
            graph.outputs[0].second.dtype != nert::DataType::Float32) {
            throw std::runtime_error("gate requires one float32 input and one float32 output");
        }
        const auto input = Read(argv[2]);
        const auto reference = Read(argv[3]);
        const auto dims = graph.inputs[0].second.dims;
        std::size_t count = 1;
        for (const auto dim : dims) {
            if (dim <= 0 || count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim))
                throw std::runtime_error("gate requires positive static input dimensions");
            count *= static_cast<std::size_t>(dim);
        }
        if (count != input.size())
            throw std::runtime_error("input element-count mismatch");
        const auto start = std::chrono::steady_clock::now();
        nert::Model model(graph, {{graph.inputs[0].first, dims}}, nert::HipDevice());
        const auto built = std::chrono::steady_clock::now();
        std::cout << "nodes=" << graph.nodes.size() << " declared_opset=" << graph.declared_opset
                  << " construct_ms=" << std::chrono::duration<double, std::milli>(built - start).count() << '\n';
        for (int run = 1; run <= 2; ++run) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto result = model.Run({{graph.inputs[0].first, input}});
            const auto t1 = std::chrono::steady_clock::now();
            const auto& got = result.outputs.at(graph.outputs[0].first);
            if (got.size() != reference.size())
                throw std::runtime_error("reference element-count mismatch");
            std::size_t over = 0;
            double max_abs = 0, max_ratio = 0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                if (!std::isfinite(got[i]) || !std::isfinite(reference[i]))
                    throw std::runtime_error("non-finite output/reference");
                const double diff = std::abs(static_cast<double>(got[i]) - reference[i]);
                const double ratio = diff / (1e-3 * (1 + std::abs(static_cast<double>(reference[i]))));
                max_abs = std::max(max_abs, diff);
                max_ratio = std::max(max_ratio, ratio);
                if (ratio > 1)
                    ++over;
            }
            std::cout << "run=" << run << " elements=" << got.size() << " over_tolerance=" << over
                      << " max_abs=" << max_abs << " max_ratio=" << max_ratio
                      << " run_ms=" << std::chrono::duration<double, std::milli>(t1 - t0).count() << std::endl;
            if (over != 0)
                return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
