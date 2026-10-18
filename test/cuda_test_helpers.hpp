// Shared CUDA test helpers: device presence, host/device float + int64 views,
// upload/download, node construction, and the bench-tolerance comparison.
//
// Included by every test/cuda_*_test.cpp so the seam-driving boilerplate
// lives in exactly one place. Everything sits in an anonymous namespace
// (internal linkage per TU — the same semantics as the per-file blocks this
// replaces). GTEST_SKIP must still fire in the TEST body (it only returns
// from its own function), so HasCudaDevice reports readiness and each TEST
// skips explicitly.

#pragma once

#include "nert/device.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool HasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count >= 1;
}

nert::TensorView HostFloat(std::vector<float>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

nert::TensorView HostInt64(std::vector<int64_t>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

// A device copy of a host buffer, released by the caller.
void* Upload(nert::Device& cuda, const std::vector<float>& host) {
    void* device = cuda.allocator().allocate(static_cast<int64_t>(host.size() * sizeof(float)));
    if (device == nullptr) {
        throw std::runtime_error("Upload: device allocate failed");
    }
    cuda.transfer().host_to_device(device, host.data(), static_cast<int64_t>(host.size() * sizeof(float)));
    return device;
}

void* UploadInt64(nert::Device& cuda, const std::vector<int64_t>& host) {
    void* device = cuda.allocator().allocate(static_cast<int64_t>(host.size() * sizeof(int64_t)));
    if (device == nullptr) {
        throw std::runtime_error("UploadInt64: device allocate failed");
    }
    cuda.transfer().host_to_device(device, host.data(), static_cast<int64_t>(host.size() * sizeof(int64_t)));
    return device;
}

std::vector<float> Download(nert::Device& cuda, void* device, std::size_t count) {
    std::vector<float> host(count, 0.0F);
    cuda.transfer().device_to_host(host.data(), device, static_cast<int64_t>(count * sizeof(float)));
    return host;
}

std::vector<int64_t> DownloadInt64(nert::Device& cuda, void* device, std::size_t count) {
    std::vector<int64_t> host(count, 0);
    cuda.transfer().device_to_host(host.data(), device, static_cast<int64_t>(count * sizeof(int64_t)));
    return host;
}

nert::TensorView HostBool(std::vector<std::uint8_t>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Bool;
    view.dims = std::move(dims);
    return view;
}

void* UploadBool(nert::Device& cuda, const std::vector<std::uint8_t>& host) {
    void* device = cuda.allocator().allocate(static_cast<int64_t>(host.size()));
    if (device == nullptr) {
        throw std::runtime_error("UploadBool: device allocate failed");
    }
    cuda.transfer().host_to_device(device, host.data(), static_cast<int64_t>(host.size()));
    return device;
}

std::vector<std::uint8_t> DownloadBool(nert::Device& cuda, void* device, std::size_t count) {
    std::vector<std::uint8_t> host(count, 0);
    cuda.transfer().device_to_host(host.data(), device, static_cast<int64_t>(count));
    return host;
}

nert::TensorView DeviceFloat(void* device, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = device;
    view.dtype = nert::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

nert::TensorView DeviceInt64(void* device, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = device;
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

nert::TensorView DeviceBool(void* device, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = device;
    view.dtype = nert::DataType::Bool;
    view.dims = std::move(dims);
    return view;
}

nert::Node MakeNode(const std::string& name, const std::string& op) {
    nert::Node node;
    node.name = name;
    node.op_type = op;
    return node;
}

// |a - b| <= 1e-3 * (1 + |b|) elementwise, the bench tolerance.
void ExpectNear(const std::vector<float>& got, const std::vector<float>& ref) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3 * (1.0 + std::fabs(ref[i]))) << "index " << i;
    }
}

} // namespace
