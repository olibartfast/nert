// CPU fallback for the CUDA-path device wrapper (G2).
//
// G3 replaces this file with the tiled CUDA implementation (compiled
// `LANGUAGE CUDA`). Until then `gemmGPU` throws so the harness reports the
// device half as SKIP while the CPU oracle half still runs and still fails
// on a genuine mismatch.

#include "cuda/gemm.hpp"

#include <stdexcept>

namespace nert_bench {

void gemmGPU(const float* /*h_A*/, const float* /*h_B*/, float* /*h_C*/, float /*alpha*/, float /*beta*/, int /*M*/,
             int /*N*/, int /*K*/) {
    throw std::runtime_error("gemmGPU: CUDA backend not enabled in this build (G2 CPU fallback)");
}

} // namespace nert_bench
