// To compile this file, pass in a macro for the compute (T2SP_S/DDOT), the size of the systolic array (TINY or LARGE), and the hardware(A10 or S10).
// And pass in a macro FPGA_EMULATOR if to use the emulator instead of FPGA hardware.

#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// The GEMM API to invoke
#include "./api.hpp"

// Useful routines from the OneMKL unit tests
#include "allocator_helper.hpp"
#include "oneapi/mkl.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include "exception_handler.hpp"

using namespace std;

template <typename T>
void test(int N, int incx, int incy) {
    vector<T, allocator_helper<T, 64>> x, y;
    T res{};
    rand_vector(x, N, incx);
    rand_vector(y, N, incy);
    auto res_ref = res;

// Create a queue bound to either the FPGA emulator or FPGA device.
#if defined(FPGA_EMULATOR)
    sycl::queue q_device(sycl::ext::intel::fpga_emulator_selector_v, fpga_tools::exception_handler);
#else
    sycl::queue q_device(sycl::ext::intel::fpga_selector_v, fpga_tools::exception_handler);
#endif

    auto done = t2sp::blas::row_major::dot(q_device, N, x.data(), incx, y.data(),
                                           incy, &res);
    done.wait();

#ifdef CHECK_CORRECTNESS
    // Call oneMKL GEMM as reference.
    sycl::queue main_queue(sycl::cpu_selector_v);
    oneapi::mkl::blas::row_major::dot(main_queue, N, x.data(), incx, y.data(),
                                      incy, &res_ref).wait();
    bool correct = check_equal_ptr(main_queue, res, *res_ref, N, std::cout); 
    assert(correct);
    std::cout << "Correct!\n";
#else
    // Get time in ns
    uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    uint64_t exec_time = end - start;
    std::cout << "Execution time in nanoseconds = " << exec_time << "\n";

    double number_ops = 2.0 * N;
    std::cout << "GFLOPs: " << number_ops / exec_time << "\n";
    std::cout << "Size of vector x: " << N << "\n";
    std::cout << "Size of vector y: " << N << "\n"; 
#endif
}

int main() {
#if defined(T2SP_SDOT)
    const auto KKK = t2sp::blas::row_major::get_systolic_array_dimensions<float>();
    int64_t n = KKK * 64 * 128;
    test<float>(n, 1, 1);
#elif defined(T2SP_DDOT)
    const auto KKK = t2sp::blas::row_major::get_systolic_array_dimensions<double>();
    int64_t n = KKK * 64 * 128;
    test<double>(n, 1, 1);
#endif
}
