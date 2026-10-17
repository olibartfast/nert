#pragma once
// Shared helpers for the device-layer test suites (OpenCL and HIP).
//
// Both suites exercise the same seam the same way: probe presence in the
// TEST body, upload fixture data, run a kernel from the device table, and
// download for comparison. Keeping that here means one behavior and no
// per-backend copies; the backend-specific files hold only fixtures.

#include "nert/device.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nert_test {

// True when the device can be constructed. Device constructors throw
// InferenceException naming the missing platform/driver; that is a SKIP,
// never a failure. Call this inside the TEST body (GTEST_SKIP depends on
// returning from the test, not a helper).
inline bool DeviceAvailable(const nert::Device& (*factory)()) {
    try {
        (void)factory();
        return true;
    } catch (const nert::InferenceException&) {
        return false;
    }
}

inline nert::TensorView HostFloat(std::vector<float>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

inline nert::TensorView HostInt64(std::vector<int64_t>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

inline void* Upload(nert::Device& device, const std::vector<float>& host) {
    const int64_t bytes = static_cast<int64_t>(host.size() * sizeof(float));
    void* buffer = device.allocator().allocate(bytes);
    if (buffer == nullptr) {
        throw std::runtime_error("Upload: device allocate failed");
    }
    device.transfer().host_to_device(buffer, host.data(), bytes);
    return buffer;
}

inline void* UploadInt64(nert::Device& device, const std::vector<int64_t>& host) {
    const int64_t bytes = static_cast<int64_t>(host.size() * sizeof(int64_t));
    void* buffer = device.allocator().allocate(bytes);
    if (buffer == nullptr) {
        throw std::runtime_error("UploadInt64: device allocate failed");
    }
    device.transfer().host_to_device(buffer, host.data(), bytes);
    return buffer;
}

inline std::vector<float> Download(nert::Device& device, void* buffer, std::size_t count) {
    std::vector<float> host(count, 0.0F);
    device.transfer().device_to_host(host.data(), buffer, static_cast<int64_t>(count * sizeof(float)));
    return host;
}

inline std::vector<int64_t> DownloadInt64(nert::Device& device, void* buffer, std::size_t count) {
    std::vector<int64_t> host(count, 0);
    device.transfer().device_to_host(host.data(), buffer, static_cast<int64_t>(count * sizeof(int64_t)));
    return host;
}

inline nert::TensorView DeviceFloat(void* buffer, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = buffer;
    view.dtype = nert::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

inline nert::TensorView DeviceInt64(void* buffer, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = buffer;
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

inline nert::Node MakeNode(const std::string& name, const std::string& op) {
    nert::Node node;
    node.name = name;
    node.op_type = op;
    return node;
}

} // namespace nert_test
