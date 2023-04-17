#include <cstdlib>
#include <iostream>

#include "sdot.large.sycl.h"
#include "parameters.h"
#include "HalideBuffer.h"

int main() {
    using Halide::Runtime::Buffer;
    constexpr int N = KKK * KK * 128;
    Buffer<float> x(N), y(N), result(1);
    for (int k = 0; k < N; k++)
        x(k) = float(std::rand()) / float(RAND_MAX) - 0.5f;
    for (int k = 0; k < N; k++)
        y(k) = float(std::rand()) / float(RAND_MAX) - 0.5f;

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

     auto exec_time = t2sp::sdot::sdot(device_selector, 1, 1, x, y, result);

     double number_ops = 2.0 * N;
     std::cout << "GFlops: " << number_ops / exec_time << "\n";
     std::cout << "Size of vector X: " << N << "\n";
     std::cout << "Size of vector Y: " << N << "\n";
     return 0;
}
