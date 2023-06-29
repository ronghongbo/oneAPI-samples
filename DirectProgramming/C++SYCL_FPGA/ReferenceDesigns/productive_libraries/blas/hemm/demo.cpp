// To compile this file, pass in a macro for the compute (T2SP_S/D/c/ZMATMUL), the size of the systolic array (TINY or LARGE), and the hardware(A10 or S10).
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
void test(oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower, int m, int n, int lda, int ldb, int ldc, T alpha, T beta) {
    vector<T, allocator_helper<T, 64>> a, b;
    vector<T, allocator_helper<T, 64>> c, c_ref;
    rand_matrix(a, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::N, left_right == oneapi::mkl::side::left ? m : n,
                                                                              left_right == oneapi::mkl::side::left ? m : n, lda);
    // set the elements on the diagonal to real numbers
    for (int i = 0; i < (left_right == oneapi::mkl::side::left ? m : n); i++) {
        a[i + i * lda] = a[i + i * lda].real();
    }

    rand_matrix(b, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::N, m, n, ldb);
    rand_matrix(c, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::N, m, n, ldc);
    c_ref = c;

// Create a queue bound to either the FPGA emulator or FPGA device.
#if defined(FPGA_EMULATOR)
    sycl::queue q_device(sycl::ext::intel::fpga_emulator_selector_v, fpga_tools::exception_handler);
#else
    sycl::queue q_device(sycl::ext::intel::fpga_selector_v, fpga_tools::exception_handler);
#endif

    sycl::event e = t2sp::blas::row_major::hemm(q_device, left_right, upper_lower, m, n, alpha, a.data(), lda,
                                                b.data(), ldb, beta, c.data(), ldc);
    e.wait();

#ifdef CHECK_CORRECTNESS
    // Call oneMKL GEMM as reference.
    sycl::queue main_queue(sycl::cpu_selector_v);
    oneapi::mkl::blas::row_major::hemm(main_queue, left_right, upper_lower, m, n, alpha, a.data(), lda,
                                       b.data(), ldb, beta, c_ref.data(), ldc).wait();
    bool correct = check_equal_matrix(c.data(), c_ref.data(), oneapi::mkl::layout::row_major, m, n, ldc,  10 * m, std::cout);
    assert(correct);
    std::cout << "Correct!\n";
#else
    // Get time in ns
    uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    uint64_t exec_time = end - start;
    std::cout << "Execution time in nanoseconds = " << exec_time << "\n";

    // TOFIX
    double number_ops;
    if ((std::is_same_v<float, T> || std::is_same_v<double, T>)) {
        // FP operations per MAD (MUL and ADD) for float and double =2
        number_ops = 2.0 * m * n * k + m * n;
    } else {
        // FP operations per MAD (MUL and ADD) for complex float and double =8:
        // Multiplying two complex numbers requires 4 rand_matrixFP MUL and 2 FP ADD
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
#if defined(T2SP_SMATMUL)
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<float>();
    int64_t m = III * II * 4;
    int64_t n = JJJ * JJ * 4;
    int64_t lda = m;
    int64_t ldb = n;
    int64_t ldc = n;
    float alpha = 2.0f;
    float beta  = 3.0f;
    test<float>(oneapi::mkl::side::L, oneapi::mkl::uplo::U, m, n, lda, ldb, ldc, alpha, beta);
#elif defined(T2SP_DMATMUL)
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<double>();
    int64_t m = III * II * 32;
    int64_t n = JJJ * JJ * 32;
    int64_t lda = m;
    int64_t ldb = n;
    int64_t ldc = n;
    double alpha = 2.0f;
    double beta = 3.0f;
    test<double>(oneapi::mkl::side::L, oneapi::mkl::uplo::U, m, n, lda, ldb, ldc, alpha, beta);
#elif defined(T2SP_CMATMUL)
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<std::complex<float>>();
    int64_t m = III * II * 32;
    int64_t n = JJJ * JJ * 32;
    int64_t lda = m;
    int64_t ldb = n;
    int64_t ldc = n;
    std::complex<float> alpha = {2.0f, -0.5f};
    std::complex<float> beta  = {3.0f, -1.5f};
    test<std::complex<float>>(oneapi::mkl::side::L, oneapi::mkl::uplo::U, m, n, lda, ldb, ldc, alpha, beta);
#else
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<std::complex<double>>();
    int64_t m = III * II * 32;
    int64_t n = JJJ * JJ * 32;
    int64_t lda = m;
    int64_t ldb = n;
    int64_t ldc = n;
    std::complex<double> alpha = {2.0f, -0.5f};
    std::complex<double> beta  = {3.0f, -1.5f};
    test<std::complex<double>>(oneapi::mkl::side::L, oneapi::mkl::uplo::U, m, n, lda, ldb, ldc, alpha, beta);
#endif
}
