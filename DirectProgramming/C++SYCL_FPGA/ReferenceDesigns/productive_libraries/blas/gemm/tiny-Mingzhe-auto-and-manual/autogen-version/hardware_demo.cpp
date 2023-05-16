#include <cstdlib>
#include <iostream>
#include <vector>

#include "interface.hpp"

int main() {
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 64;

    constexpr int lda = 64, ldb = 64, ldc = 64;
    constexpr auto alpha = 2.0f;
    constexpr auto beta = 3.0f;

    std::vector<float> a(M * lda);
    std::vector<float> b(K * ldb);
    std::vector<float> c(M * ldc);
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            a[k + i * lda] = float(std::rand()) / RAND_MAX - 0.5f;

    for (size_t k = 0; k < K; k++)
        for (size_t j = 0; j < N; j++)
            b[j + k * ldb] = float(std::rand()) / RAND_MAX - 0.5f;

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            c[j + i * ldc] = float(std::rand()) / RAND_MAX - 0.5f;

    auto exec_time = t2sp::gemm(false, false, M, N, K, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);

    double number_ops = 2.0 * M * N * K + M * N;
    std::cout << "GFlops: " << number_ops / exec_time << "\n";
    std::cout << "Size of matrix A: " << M << " * " << K << "\n";
    std::cout << "Size of matrix B: " << K << " * " << N << "\n";
    std::cout << "Size of matrix C: " << M << " * " << N << "\n";
    return 0;
}
