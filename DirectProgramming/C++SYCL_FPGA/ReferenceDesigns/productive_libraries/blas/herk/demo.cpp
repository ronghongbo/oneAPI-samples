// To compile this file, pass in a macro for the compute (T2SP_S/D/c/ZMATMUL), the size of the systolic array (TINY or LARGE), and the hardware(A10 or S10).
// And pass in a macro FPGA_EMULATOR if to use the emulator instead of FPGA hardware.

#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// The HERK API to invoke
#include "./api.hpp"

// Useful routines from the OneMKL unit tests
#include "allocator_helper.hpp"
#include "oneapi/mkl.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include "exception_handler.hpp"

using namespace std;

template <typename T, typename T_REAL>
int test(oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, int n, int k, int lda, int ldc, T_REAL alpha, T_REAL beta) {
    vector<T, allocator_helper<T, 64>> a;
    vector<T, allocator_helper<T, 64>> c, c_ref;
    rand_matrix(a, oneapi::mkl::layout::row_major, trans, n, k, lda);
    rand_matrix(c, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::nontrans, n, n, ldc);
    c_ref = c;

// Create a queue bound to either the FPGA emulator or FPGA device.
#if defined(FPGA_EMULATOR)
    sycl::queue q_device(sycl::ext::intel::fpga_emulator_selector_v, fpga_tools::exception_handler);
#else
    sycl::queue q_device(sycl::ext::intel::fpga_selector_v, fpga_tools::exception_handler);
#endif

    sycl::event e = t2sp::blas::row_major::herk(q_device, upper_lower, trans, n, k,
                                                alpha, a.data(), lda, beta, c.data(), ldc);
    e.wait();

#ifdef CHECK_CORRECTNESS
    // Call oneMKL HERK as reference.
    sycl::queue main_queue(sycl::cpu_selector_v);
    oneapi::mkl::blas::row_major::herk(main_queue, upper_lower, trans, n, k,
                                       alpha, a.data(), lda, beta, c_ref.data(), ldc).wait();
    bool correct = check_equal_matrix(c, c_ref, oneapi::mkl::layout::row_major, n, n, ldc, 10 * std::max(n, k), std::cout);
    assert(correct);
    std::cout << "Correct!\n";
#else
    // Get time in ns
    uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    uint64_t exec_time = end - start;
    std::cout << "Execution time in nanoseconds = " << exec_time << "\n";

    // TOFIX: the number of operations below
    double number_ops;
    if ((std::is_same_v<float, T> || std::is_same_v<double, T>)) {
        // FP operations per MAD (MUL and ADD) for float and double =2
        number_ops = 2.0 * m * n * k + m * n;
    } else {
        // FP operations per MAD (MUL and ADD) for complex float and double =8:
        // Multiplying two complex numbers requires 4 FP MUL and 2 FP ADD
        // Adding two complex numbers requires 2 FP ADD
        number_ops = 8.0 * m * n * k + 2.0 * m * n;
    }
    std::cout << "GFLOPs: " << number_ops / exec_time << "\n";
    std::cout << "Size of matrix a: " << m << " * " << k << "\n";
    std::cout << "Size of matrix b: " << k << " * " << n << "\n";
    std::cout << "Size of matrix c: " << m << " * " << n << "\n";
#endif
}

int main() {
#if defined(PREFIX_C)
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<std::complex<float>>();
    int64_t n = III * II * 4;
    int64_t k = KKK * KK * 4;
    int64_t lda = k;
    int64_t ldc = n;
    float alpha = 2.0f;
    float beta  = 3.0f;
    test<std::complex<float>, float>(oneapi::mkl::uplo::U, oneapi::mkl::transpose::N, n, k, lda, ldc, alpha, beta);
#else
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<std::complex<double>>();
    int64_t n = III * II * 4;
    int64_t k = KKK * KK * 4;
    int64_t lda = k;
    int64_t ldc = n;
    double alpha = 2.0f;
    double beta  = 3.0f;
    test<std::complex<double>, double>(oneapi::mkl::uplo::U, oneapi::mkl::transpose::N, n, k, lda, ldc, alpha, beta);
#endif
}
