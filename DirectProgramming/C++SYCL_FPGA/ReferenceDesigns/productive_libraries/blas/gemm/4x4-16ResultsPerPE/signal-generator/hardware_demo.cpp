#include <cstdlib>
#include <iostream>
#include <vector>

#include "./sgemm.sycl.h"
#include "parameters.h"
#include "HalideBuffer.h"

int main() {
    using Halide::Runtime::Buffer;
    constexpr int M = III * II * 32;
    constexpr int N = JJJ * JJ * 32;
    constexpr int K = KKK * KK * 32;

    constexpr auto alpha = 2.0f;
    constexpr auto beta = 3.0f; 

    Buffer<float> a(K, M);
    Buffer<float> b(N, K);
    Buffer<float> c(N, M);
    Buffer<float> out(JJJ, JJ, II, III, 32, 32);
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            a(k, i) = float(std::rand()) / RAND_MAX - 0.5f; 

    for (size_t k = 0; k < K; k++)
        for (size_t j = 0; j < N; j++)
            b(j, k) = float(std::rand()) / RAND_MAX - 0.5f; 

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            c(j, i) = float(std::rand()) / RAND_MAX - 0.5f;

#if defined(SYCL_LANGUAGE_VERSION) && SYCL_LANGUAGE_VERSION >= 202001
#if defined(FPGA_EMULATOR) || defined(TEST)
    auto device_selector = +[](const sycl::device &device){
        return device.get_platform().get_info<sycl::info::platform::name>()
            == sycl::ext::intel::EMULATION_PLATFORM_NAME ? 10000 : -1;
    };
#else
    auto device_selector = +[](const sycl::device &device){
        return device.get_platform().get_info<sycl::info::platform::name>()
            == sycl::ext::intel::HARDWARE_PLATFORM_NAME ? 10000 : -1;
    };
#endif
#else
#if defined(FPGA_EMULATOR) || defined(TEST)
    sycl::ext::intel::fpga_emulator_selector device_selector{};
#else
    sycl::ext::intel::fpga_selector device_selector{};
#endif
#endif

    auto exec_time = t2sp::sgemm::sgemm(device_selector, false, false, alpha, beta, a, b, c, out); 

    double number_ops = 2.0 * M * N * K + M * N;
    std::cout << "GFlops: " << number_ops / exec_time << "\n";
    std::cout << "Size of matrix A: " << M << " * " << K << "\n";
    std::cout << "Size of matrix B: " << K << " * " << N << "\n";
    std::cout << "Size of matrix C: " << M << " * " << N << "\n";
    return 0;
}
