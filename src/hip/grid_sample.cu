// Hipified from src/cuda/grid_sample.cu via hipify-perl, hand-reviewed.
// GridSample on HIP: sample a 4-D float32 input at normalized grid
// positions.
//
// Mirrors the CPU oracle exactly: bilinear/nearest mode, zeros/border/
// reflection padding, align_corners 0/1, all coordinate math in double.
// One thread per (n, ho, wo) position computes the source coordinate and
// weights once, then loops over channels (the oracle's structure).

#include "nert/device.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
namespace {

void CheckHip(hipError_t status, const char* what, const std::string& context) {
    if (status != hipSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + hipGetErrorString(status));
    }
}

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback,
                            const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be a string");
            }
            return *value;
        }
    }
    return fallback;
}

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be an int");
            }
            return *value;
        }
    }
    return fallback;
}

// Reflect x about [lo, hi] (the border coordinates), then clamp. Matches
// the oracle exactly (double throughout).
__device__ double Reflect(double x, double lo, double hi) {
    if (hi == lo) {
        return lo;
    }
    const double range = hi - lo;
    double d = fabs(x - lo);
    const double flips = floor(d / range);
    const double extra = d - flips * range;
    return fmod(flips, 2.0) == 0.0 ? lo + extra : hi - extra;
}

// Normalized coordinate -> source pixel coordinate on an axis of `len`
// pixels, with the padding mode applied to the continuous coordinate.
// mode: 0 = zeros, 1 = border, 2 = reflection.
__device__ double SourceCoordinate(double g, int64_t len, bool alignCorners, int padding) {
    const double size = static_cast<double>(len);
    double x = alignCorners ? (g + 1.0) / 2.0 * (size - 1.0) : ((g + 1.0) * size - 1.0) / 2.0;
    if (padding == 2) {
        x = alignCorners ? Reflect(x, 0.0, size - 1.0) : Reflect(x, -0.5, size - 0.5);
        x = fmin(fmax(x, 0.0), size - 1.0);
    } else if (padding == 1) {
        x = fmin(fmax(x, 0.0), size - 1.0);
    }
    return x;
}

__global__ void GridSampleKernel(const float* xp, const float* gp, float* yp, int64_t n, int64_t c, int64_t h,
                                 int64_t w, int64_t ho, int64_t wo, bool alignCorners, int padding, bool nearest) {
    const int64_t positions = n * ho * wo;
    for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x; p < positions; p += blockDim.x * gridDim.x) {
        const int64_t j = p % wo;
        const int64_t i = (p / wo) % ho;
        const int64_t b = p / (ho * wo);
        const float* g = gp + ((b * ho + i) * wo + j) * 2;
        const double px = SourceCoordinate(static_cast<double>(g[0]), w, alignCorners, padding);
        const double py = SourceCoordinate(static_cast<double>(g[1]), h, alignCorners, padding);
        for (int64_t ch = 0; ch < c; ++ch) {
            const float* plane = xp + (b * c + ch) * h * w;
            double value = 0.0;
            if (nearest) {
                const int64_t xx = static_cast<int64_t>(nearbyint(px));
                const int64_t yy = static_cast<int64_t>(nearbyint(py));
                if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                    value = static_cast<double>(plane[yy * w + xx]);
                }
            } else {
                const double fx = floor(px);
                const double fy = floor(py);
                const int64_t x0 = static_cast<int64_t>(fx);
                const int64_t y0 = static_cast<int64_t>(fy);
                const double wx = px - fx;
                const double wy = py - fy;
                double v00 = 0.0, v01 = 0.0, v10 = 0.0, v11 = 0.0;
                if (y0 >= 0 && y0 < h && x0 >= 0 && x0 < w) {
                    v00 = static_cast<double>(plane[y0 * w + x0]);
                }
                if (y0 >= 0 && y0 < h && x0 + 1 >= 0 && x0 + 1 < w) {
                    v01 = static_cast<double>(plane[y0 * w + x0 + 1]);
                }
                if (y0 + 1 >= 0 && y0 + 1 < h && x0 >= 0 && x0 < w) {
                    v10 = static_cast<double>(plane[(y0 + 1) * w + x0]);
                }
                if (y0 + 1 >= 0 && y0 + 1 < h && x0 + 1 >= 0 && x0 + 1 < w) {
                    v11 = static_cast<double>(plane[(y0 + 1) * w + x0 + 1]);
                }
                value = v00 * (1.0 - wx) * (1.0 - wy) + v01 * wx * (1.0 - wy) + v10 * (1.0 - wx) * wy + v11 * wx * wy;
            }
            yp[((b * c + ch) * ho + i) * wo + j] = static_cast<float>(value);
        }
    }
}

} // namespace

void GridSample(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "GridSample (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const std::string mode = StringAttribute(node, "mode", "bilinear", context);
    const std::string paddingName = StringAttribute(node, "padding_mode", "zeros", context);
    const int64_t alignAttr = IntAttribute(node, "align_corners", 0, context);
    if (mode != "bilinear" && mode != "nearest") {
        throw InferenceException(context + ": unsupported mode '" + mode + "'");
    }
    if (paddingName != "zeros" && paddingName != "border" && paddingName != "reflection") {
        throw InferenceException(context + ": unsupported padding_mode '" + paddingName + "'");
    }
    if (alignAttr != 0 && alignAttr != 1) {
        throw InferenceException(context + ": align_corners must be 0 or 1");
    }
    const bool alignCorners = alignAttr == 1;
    const int padding = paddingName == "zeros" ? 0 : paddingName == "border" ? 1 : 2;

    const TensorView& x = inputs[0];
    const TensorView& grid = inputs[1];
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || grid.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() != 4 || grid.dims.size() != 4 || grid.dims[3] != 2) {
        throw InferenceException(context + ": expected X [N,C,H,W] and grid [N,Ho,Wo,2]");
    }
    const int64_t n = x.dims[0];
    const int64_t c = x.dims[1];
    const int64_t h = x.dims[2];
    const int64_t w = x.dims[3];
    const int64_t ho = grid.dims[1];
    const int64_t wo = grid.dims[2];
    if (grid.dims[0] != n) {
        throw InferenceException(context + ": batch sizes of X and grid differ");
    }
    if (out.dims != std::vector<int64_t>({n, c, ho, wo})) {
        throw InferenceException(context + ": output shape does not match the operands");
    }
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || ho <= 0 || wo <= 0) {
        throw InferenceException(context + ": non-positive dimension");
    }
    if (x.data == nullptr || grid.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const int64_t positions = n * ho * wo;
    if (positions == 0) {
        return;
    }
    const int block = 256;
    const int gridSize = static_cast<int>((positions + block - 1) / block);
    GridSampleKernel<<<gridSize, block>>>(static_cast<const float*>(x.data), static_cast<const float*>(grid.data),
                                          static_cast<float*>(out.data), n, c, h, w, ho, wo, alignCorners, padding,
                                          mode == "nearest");
    CheckHip(hipGetLastError(), "GridSampleKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
