// Generic full-model gate runner for NERT (scratch harness, never committed).
//
// Usage:
//   nert_gate <model.onnx> <cpu|cuda|opencl> <out-prefix> [<name>=<path.bin> ...]
//
// Loads the graph, binds each graph input from a raw little-endian binary file
// (float32, or int64 when the graph input dtype is Int64), runs the whole model
// with RunTyped on the chosen device, and writes every declared output as raw
// float32 / int64 to <out-prefix>_<name>.bin. '/' in tensor names is replaced
// with '_' in file names. Prints one line per output (name, dtype, element
// count, bytes written) plus wall time and the dispatched node count.
//
// Includes only the public engine headers; links against one libnert.a.
#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/model_loader.hpp"

// Backend availability is fixed at compile time: each libnert.a defines only
// the devices its build enabled, and a strong reference to an absent symbol
// cannot link (a weak reference would never pull the member out of a static
// archive). The per-backend build scripts pass -DNERT_GATE_ENABLE_CUDA /
// -DNERT_GATE_ENABLE_OPENCL when the archive provides that device; the CPU
// device exists in every archive, so no flag is needed for it.
#if defined(NERT_GATE_ENABLE_CUDA)
#define NERT_GATE_HAS_CUDA 1
#else
#define NERT_GATE_HAS_CUDA 0
#endif
#if defined(NERT_GATE_ENABLE_OPENCL)
#define NERT_GATE_HAS_OPENCL 1
#else
#define NERT_GATE_HAS_OPENCL 0
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Sanitize a tensor name for use in a file name.
std::string SanitizeName(const std::string& name) {
    std::string out = name;
    for (char& ch : out) {
        if (ch == '/') {
            ch = '_';
        }
    }
    return out;
}

int64_t ElementCount(const std::vector<int64_t>& dims) {
    int64_t count = 1;
    for (int64_t d : dims) {
        count *= d;
    }
    return count;
}

// The concrete dims a graph input is run at: declared dims with every
// dynamic/negative dim resolved to 1.
std::vector<int64_t> ConcreteDims(const std::vector<int64_t>& declared) {
    std::vector<int64_t> dims = declared;
    for (int64_t& d : dims) {
        if (d < 0) {
            d = 1;
        }
    }
    return dims;
}

std::string DtypeName(nert::DataType dtype) {
    switch (dtype) {
    case nert::DataType::Float32: return "float32";
    case nert::DataType::Int64: return "int64";
    case nert::DataType::Bool: return "bool";
    default: return "unknown";
    }
}

// Read `count` elements of `elem_size` bytes from `path` into a byte buffer.
// The file length must match exactly. Host is little-endian x86-64, so the raw
// bytes are the little-endian payload the task specifies.
std::vector<uint8_t> ReadRawFile(const std::string& path, int64_t count, size_t elem_size) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        throw std::runtime_error("cannot open input file " + path);
    }
    const size_t want = static_cast<size_t>(count) * elem_size;
    std::vector<uint8_t> bytes(want);
    const size_t got = std::fread(bytes.data(), 1, want, f);
    std::fclose(f);
    if (got != want) {
        throw std::runtime_error("input file " + path + " has " + std::to_string(got)
                                 + " bytes, expected " + std::to_string(want));
    }
    return bytes;
}

const nert::Device& SelectDevice(const std::string& which) {
    if (which == "cpu") {
        return nert::CpuDevice();
    }
    if (which == "cuda") {
#if NERT_GATE_HAS_CUDA
        return nert::CudaDevice();
#else
        (void)which;
        throw std::runtime_error(
            "this nert_gate binary was built without the CUDA backend; rebuild with "
            "-DNERT_GATE_ENABLE_CUDA against a libnert.a that has nert::CudaDevice");
#endif
    }
    if (which == "opencl") {
#if NERT_GATE_HAS_OPENCL
        return nert::OpenCLDevice();
#else
        (void)which;
        throw std::runtime_error(
            "this nert_gate binary was built without the OpenCL backend; rebuild with "
            "-DNERT_GATE_ENABLE_OPENCL against a libnert.a that has nert::OpenCLDevice");
#endif
    }
    throw std::runtime_error("unknown device '" + which + "' (expected cpu|cuda|opencl)");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: nert_gate <model.onnx> <cpu|cuda|opencl> <out-prefix>"
                     " [<name>=<path.bin> ...]\n";
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string device_name = argv[2];
    const std::string out_prefix = argv[3];

    std::map<std::string, std::string> input_files;
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        const size_t eq = arg.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == arg.size()) {
            std::cerr << "bad input argument '" << arg << "' (expected <name>=<path.bin>)\n";
            return 2;
        }
        input_files[arg.substr(0, eq)] = arg.substr(eq + 1);
    }

    try {
        const nert::Graph graph = nert::LoadGraphFromFile(model_path);

        // Concrete input dims and the ShapeMap the executor plans from.
        nert::ShapeMap input_dims;
        for (const auto& entry : graph.inputs) {
            input_dims[entry.first] = ConcreteDims(entry.second.dims);
        }

        // Bind every graph input from its file. The byte count must match the
        // element count implied by the concrete dims.
        nert::HostTensorMap inputs;
        for (const auto& entry : graph.inputs) {
            const std::string& name = entry.first;
            const nert::TensorInfo& info = entry.second;
            const auto it = input_files.find(name);
            if (it == input_files.end()) {
                throw std::runtime_error("missing input binding for graph input '" + name + "'");
            }
            const std::vector<int64_t> dims = input_dims[name];
            const int64_t count = ElementCount(dims);
            nert::HostTensor tensor;
            tensor.dims = dims;
            if (info.dtype == nert::DataType::Float32) {
                tensor.dtype = nert::DataType::Float32;
                tensor.bytes = ReadRawFile(it->second, count, sizeof(float));
            } else if (info.dtype == nert::DataType::Int64) {
                tensor.dtype = nert::DataType::Int64;
                tensor.bytes = ReadRawFile(it->second, count, sizeof(int64_t));
            } else if (info.dtype == nert::DataType::Bool) {
                throw std::runtime_error("graph input '" + name
                                         + "' is Bool; Bool inputs are not supported by this gate");
            } else {
                throw std::runtime_error("graph input '" + name + "' has unsupported dtype");
            }
            std::cout << "input " << name << " dtype=" << DtypeName(tensor.dtype) << " dims=";
            for (size_t i = 0; i < dims.size(); ++i) {
                std::cout << (i > 0 ? "," : "") << dims[i];
            }
            std::cout << " file=" << it->second << "\n";
            inputs[name] = std::move(tensor);
        }

        const nert::Device& device = SelectDevice(device_name);
        const auto t0 = std::chrono::steady_clock::now();
        nert::Model model(graph, input_dims, device);
        const auto t1 = std::chrono::steady_clock::now();
        nert::HostTensorMap outputs = model.RunTyped(inputs);
        const auto t2 = std::chrono::steady_clock::now();

        for (const auto& entry : outputs) {
            const std::string& name = entry.first;
            const nert::HostTensor& tensor = entry.second;
            const std::string file = out_prefix + "_" + SanitizeName(name) + ".bin";
            std::FILE* f = std::fopen(file.c_str(), "wb");
            if (f == nullptr) {
                throw std::runtime_error("cannot open output file " + file);
            }
            const size_t written = std::fwrite(tensor.bytes.data(), 1, tensor.bytes.size(), f);
            std::fclose(f);
            if (written != tensor.bytes.size()) {
                throw std::runtime_error("short write to " + file);
            }
            std::cout << "output " << name << " dtype=" << DtypeName(tensor.dtype)
                      << " elems=" << ElementCount(tensor.dims)
                      << " bytes=" << tensor.bytes.size() << " file=" << file << "\n";
        }

        const double construct_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double run_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        const double total_ms =
            std::chrono::duration<double, std::milli>(t2 - t0).count();
        std::cout << "device=" << device.name() << " construct_ms=" << construct_ms
                  << " run_ms=" << run_ms << " total_ms=" << total_ms
                  << " nodes_dispatched=" << graph.nodes.size() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "nert_gate: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
