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
void test(oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          int64_t m, int64_t n, int64_t k, T alpha, int64_t lda, int64_t ldb, T beta, int64_t ldc) {
    vector<T, allocator_helper<T, 64>> a, b;
    vector<T, allocator_helper<T, 64>> c, c_ref;
    rand_matrix(a, oneapi::mkl::layout::row_major, transa, m, k, lda);
    rand_matrix(b, oneapi::mkl::layout::row_major, transb, k, n, ldb);
    rand_matrix(c, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::nontrans, m, n, ldc);
    c_ref = c;

    // Create a queue on the FPGA device.
    sycl::queue q_device(sycl::ext::intel::fpga_selector_v, fpga_tools::exception_handler, sycl::property::queue::enable_profiling());

    sycl::event e = t2sp::blas::row_major::gemm(q_device, transa, transb, m, n, k, alpha, a.data(), lda,
                                                b.data(), ldb, beta, c.data(), ldc);
    e.wait();

    // Statistics for performance measurement
    uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    uint64_t exec_time = end - start;

    double multiplications_in_product, additions_in_product, multiplications_in_sum, additions_in_sum, flops_per_multiplication, flops_per_addition, total_flops, total_data, bytes_per_data, total_bytes;
    // m*n results in computing product=op(A)*op(B), each result is reduced from k number of multiplications and additions.
    multiplications_in_product = m * n * k;
    additions_in_product = m * n * k;
    // m*n results in computing alpha*product + beta*C, each is reduced from two multiplications and one addition.
    multiplications_in_sum = 2.0 * m * n;
    additions_in_sum = m * n;
    if ((std::is_same_v<float, T> || std::is_same_v<double, T>)) {
        // For float and double, 1 multiplication/addition is 1 FP operation (of single- or double-precision)
        flops_per_multiplication = 1;
        flops_per_addition = 1;
    } else {
        // For complex float and double, 1 multiplication of two complex numbers requires 4 FP MUL and 2 FP ADD operations (of single- or double-precision)
        // 1 addition of two complex numbers requires 2 FP ADD operations (of single- or double-precision)
        flops_per_multiplication = 6;
        flops_per_addition = 2;
    }
    total_flops = multiplications_in_product * flops_per_multiplication + additions_in_product * flops_per_addition +
                  multiplications_in_sum     * flops_per_multiplication + additions_in_sum     * flops_per_addition;

    total_data = m * k + k * n + 2.0 * m * n; // data accessed from op(A), op(B), original C, and final C
    bytes_per_data = sizeof(T);
    total_bytes = total_data * bytes_per_data;
    std::cout << "FP operations: " << total_flops << "\n";
    std::cout << "Execution time: " << exec_time << " ns\n";
    std::cout << "GFLOPs: " << total_flops / exec_time << "\n";
    std::cout << "Memory bytes: " << total_bytes << "\n";
    std::cout << "Size of matrix a: " << m << " * " << k << "\n";
    std::cout << "Size of matrix b: " << k << " * " << n << "\n";
    std::cout << "Size of matrix c: " << m << " * " << n << "\n";
}

int main() {
#if defined(PREFIX_S)
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<float>();
    int64_t m = III * II * 32;
    int64_t n = JJJ * JJ * 32;
    int64_t k = KKK * KK * 32;
    int64_t lda = k;
    int64_t ldb = k;
    int64_t ldc = n;
    float alpha = 2.0f;
    float beta  = 3.0f;
    test<float>(oneapi::mkl::transpose::N, oneapi::mkl::transpose::T, m, n, k, alpha, lda, ldb, beta, ldc);
#elif defined(PREFIX_D)
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<double>();
    int64_t m = III * II * 32;
    int64_t n = JJJ * JJ * 32;
    int64_t k = KKK * KK * 32;
    int64_t lda = k;
    int64_t ldb = k;
    int64_t ldc = n;
    double alpha = 2.0f;
    double beta = 3.0f;
    test<double>(oneapi::mkl::transpose::N, oneapi::mkl::transpose::T, m, n, k, alpha, lda, ldb, beta, ldc);
#elif defined(PREFIX_C)
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<std::complex<float>>();
    int64_t m = III * II * 32;
    int64_t n = JJJ * JJ * 32;
    int64_t k = KKK * KK * 32;
    int64_t lda = k;
    int64_t ldb = k;
    int64_t ldc = n;
    std::complex<float> alpha = {2.0f, -0.5f};
    std::complex<float> beta  = {3.0f, -1.5f};
    test<std::complex<float>>(oneapi::mkl::transpose::N, oneapi::mkl::transpose::T, m, n, k, alpha, lda, ldb, beta, ldc);
#else
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<std::complex<double>>();
    int64_t m = III * II * 32;
    int64_t n = JJJ * JJ * 32;
    int64_t k = KKK * KK * 32;
    int64_t lda = k;
    int64_t ldb = k;
    int64_t ldc = n;
    std::complex<double> alpha = {2.0f, -0.5f};
    std::complex<double> beta  = {3.0f, -1.5f};
    test<std::complex<double>>(oneapi::mkl::transpose::N, oneapi::mkl::transpose::T, m, n, k, alpha, lda, ldb, beta, ldc);
#endif
}
