// To compile this file, pass in a macro for the compute (T2SP_S/D/C/ZMATMUL), the size of the systolic array (TINY or LARGE), and the hardware(A10 or S10).
// And pass in a macro FPGA_EMULATOR if to use the emulator instead of FPGA hardware.

#include <cstdlib>
#include <iostream>

// The programming interface of the reconfigurable matrix multiplication.
#include "reconfigurable_matmul/api.hpp"

CONST_TYPE get_random() {
#if defined(T2SP_SMATMUL) || defined(T2SP_DMATMUL)
    return (CONST_TYPE) (std::rand() / RAND_MAX - 0.5f);
#else
    return (CONST_TYPE) { std::rand() / RAND_MAX - 0.5f, std::rand() / RAND_MAX - 0.5f };
#endif
}

int main() {
    constexpr int64_t M = III * II * 32;
    constexpr int64_t N = JJJ * JJ * 32;
    constexpr int64_t K = KKK * KK * 32;
    constexpr int64_t lda = K;
    constexpr int64_t ldb = N;
    constexpr int64_t ldc = N;

    constexpr CONST_TYPE alpha = 2.0f;
    constexpr CONST_TYPE beta = 3.0f;

    CONST_TYPE a[M][K], b[K][N], c[M][N];
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            a[i][k] = get_random();

    for (size_t k = 0; k < K; k++)
        for (size_t j = 0; j < N; j++)
            b[k][j] = get_random();

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            c[i][j] = get_random();

    auto exec_time = t2sp::blas_matmul(false, false, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc);

#if defined(T2SP_SMATMUL) || defined(T2SP_DMATMUL)
    // FP operations per MAD (MUL and ADD) for float and double =2
    double number_ops = 2.0 * M * N * K + M * N;
#else
    // FP operations per MAD (MUL and ADD) for complex float and double =8:
    // Multiplying two complex numbers requires 4 FP MUL and 2 FP ADD
    // Adding two complex numbers requires 2 FP ADD
    double number_ops = 8.0 * M * N * K + 2.0 * M * N;
#endif
    std::cout << "GFLOPs: " << number_ops / exec_time << "\n";
    std::cout << "Size of matrix A: " << M << " * " << K << "\n";
    std::cout << "Size of matrix B: " << K << " * " << N << "\n";
    std::cout << "Size of matrix C: " << M << " * " << N << "\n";
    return 0;
}
