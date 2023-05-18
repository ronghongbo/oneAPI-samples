#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "gemm.tiny.h"

int main(int argc, char *argv[]) {
#if defined(FPGA_EMULATOR)
    std::cout << "USING FPGA EMULATOR" << std::endl;
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    std::cout << "USING FPGA HARDWARE" << std::endl;
    sycl::ext::intel::fpga_selector device_selector;
#endif
    sycl::queue q(device_selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling());
    std::cout << "Device name: "
              << q.get_device().get_info<sycl::info::device::name>().c_str()
              << std::endl;

    constexpr size_t I = 4;
    constexpr size_t J = 4;
    constexpr size_t K = 4;
    constexpr size_t TOTAL_I = III * II * I;
    constexpr size_t TOTAL_J = JJJ * JJ * J;
    constexpr size_t TOTAL_K = KKK * KK * K;
    constexpr size_t num_elem_A = TOTAL_K * TOTAL_I;
    constexpr size_t num_elem_B = TOTAL_J * TOTAL_K;
    constexpr size_t num_elem_C = TOTAL_J * TOTAL_I;

    float *A = (float *)malloc(num_elem_A * sizeof(float));
    float *B = (float *)malloc(num_elem_B * sizeof(float));
    float *C = (float *)malloc(num_elem_C * sizeof(float));
    float *result = (float *)malloc(num_elem_C * sizeof(float));

    // Generate the random input matrices
    for (size_t i = 0; i < TOTAL_I; i++) {
        for (size_t k = 0; k < TOTAL_K; k++) {
            A[k + i * TOTAL_K] = random();
        }
    }
    for (size_t j = 0; j < TOTAL_J; j++) {
        for (size_t k = 0; k < TOTAL_K; k++) {
            B[j + k * TOTAL_J] = A[k + j * TOTAL_K];
        }
    }

    gemm(A, B, C, result, 1, 1, I, J, K, q);

    bool passed = true;
    for (size_t i = 0; i < TOTAL_I; i++) {
         for (size_t j = 0; j < TOTAL_J; j++) {
             float golden = 0.0f;
             for (size_t k = 0; k < TOTAL_K; k++) {
                 golden += A[k + i * TOTAL_K] * B[j + k * TOTAL_J];
             }
             passed &= fabs(golden - C[j + i * TOTAL_J]) < 0.005 *
             fabs(golden);
         }
    }

    if (passed) {
        printf("[PASSED]\n");
    } else {
        printf("[FAILED]\n");
    }
    free(A);
    free(B);
    free(C);
}
