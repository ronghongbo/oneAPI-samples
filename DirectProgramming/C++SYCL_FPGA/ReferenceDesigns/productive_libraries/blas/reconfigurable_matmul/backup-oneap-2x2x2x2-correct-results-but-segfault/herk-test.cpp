#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <sycl/ext/intel/fpga_device_selector.hpp>
#include "mkl_cblas.h"
#include "oneapi/mkl.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include "./api.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

sycl::device d{sycl::cpu_selector_v};
std::vector<sycl::device*> devices{&d};

namespace {

template <typename fp, typename fp_scalar>
int test(device* dev, oneapi::mkl::layout layout, oneapi::mkl::uplo upper_lower,
         oneapi::mkl::transpose trans, int n, int k, int lda, int ldc, fp_scalar alpha,
         fp_scalar beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {https://oneapi-src.github.io/oneMKL/domains/blas/hemm.ht
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during HERK:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    queue fpga_queue(sycl::ext::intel::fpga_emulator_selector_v, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> A(ua), C(ua);
    rand_matrix(A, layout, trans, n, k, lda);

    rand_matrix(C, layout, oneapi::mkl::transpose::nontrans, n, n, ldc);
    // set the elements on the diagonal to real numbers
    for (int i = 0; i < n; i++) {
        C[i + i * lda] = C[i + i * lda].real();
    }

    auto C_ref = C;

    // For debugging
    bool TransposeA = false;
    int A_extent_0 = k;
    int A_extent_1 = n;
    int C_extent_0 = n;
    int C_extent_1 = n;

    #define KKK         2
    #define JJJ         2
    #define III         2
    #define JJ          2
    #define II          2
    #define KK          2

    #define C_COLS             (C_extent_0)
    #define C_ROWS             (C_extent_1)
    #define A_COLS             (A_extent_0)
    #define A_ROWS             (A_extent_1)
    #define select(c, x, y)    ((c) ? (x) : (y))
    #define REDUCTIOIN_LEN     select(TransposeA, A_ROWS, A_COLS)


    #define I                  ((C_ROWS    + (III * II - 1)) / (III * II))
    #define J                  ((C_COLS    + (JJJ * JJ - 1)) / (JJJ * JJ))
    #define K                  ((REDUCTIOIN_LEN + (KKK * KK - 1)) / (KKK * KK))

    #define C_row_idx          (iii + III * ii + III * II * i)
    #define C_col_idx          (jjj + JJJ * jj + JJJ * JJ * j)
    #define reduction_idx      (kkk + KKK * kk + KKK * KK * k)

    for (int i = 0; i < I; i++) {
        for (int j = i; j < J; j++){
            std::complex<float> sum[II][JJ][III][JJJ];
            for (int ii = 0; ii < 2; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int iii = 0; iii < III; iii++){
                        for (int jjj = 0; jjj < JJJ; jjj++) {
                            sum[ii][jj][iii][jjj] = {0.0f, 0.0f};
                        }
                    }
                }
            }
            for (int k = 0; k < K; k++) {
                for (int kk = 0; kk < 2; kk++) {
                  for (int iii = 0; iii < III; iii++){
                    for (int ii = 0; ii < 2; ii++) {
                        for (int jj = 0; jj < 2; jj++) {
                                for (int jjj = 0; jjj < JJJ; jjj++) {
                                    for (int kkk = 0; kkk < KKK; kkk++) {
                                      std::complex<float> valA = A[reduction_idx + C_row_idx*lda];
                                      std::complex<float> valAH = std::conj(A[reduction_idx + C_col_idx*lda]);
                                      sum[ii][jj][iii][jjj] += valA * valAH;
                                      if (k == K - 1 && kk == KK - 1 && kkk == KKK -1 ) {
                                          sum[ii][jj][iii][jjj] = alpha*sum[ii][jj][iii][jjj] + beta * C[C_col_idx + C_row_idx * ldc];
                                          // Ignore the values that are really under the diagonal.
                                          if (C_row_idx <= C_col_idx) {
                                              printf("OrigResult[%d, %d](%f, %f)\n",
                                                  C_row_idx, C_col_idx,
                                                  sum[ii][jj][iii][jjj].real(), sum[ii][jj][iii][jjj].imag());
                                          }
                                      }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < I; i++) {
        for (int j = i; j < J; j++){
            std::complex<float> sum[II][JJ][III][JJJ];
            for (int ii = 0; ii < 2; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int iii = 0; iii < III; iii++){
                        for (int jjj = 0; jjj < JJJ; jjj++) {
                            sum[ii][jj][iii][jjj] = {0.0f, 0.0f};
                        }
                    }
                }
            }
            for (int k = 0; k < K; k++) {
                for (int kk = 0; kk < 2; kk++) {
                    for (int ii = 0; ii < 2; ii++) {
                        for (int jj = 0; jj < 2; jj++) {
                            for (int iii = 0; iii < III; iii++){
                                for (int jjj = 0; jjj < JJJ; jjj++) {
                                    for (int kkk = 0; kkk < KKK; kkk++) {
                                      std::complex<float> valA = A[reduction_idx + C_row_idx*lda];
                                      std::complex<float> valAH = std::conj(A[reduction_idx + C_col_idx*lda]);
                                      sum[ii][jj][iii][jjj] += valA * valAH;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (int iii = 0; iii < III; iii++){
                for (int ii = 0; ii < 2; ii++) {
                    for (int jj = 0; jj < 2; jj++) {
                        for (int jjj = 0; jjj < JJJ; jjj++) {
                                      printf("OrigProduct(%d, %d) Product=(%f, %f)\n",
                                              C_row_idx, C_col_idx,
                                              sum[ii][jj][iii][jjj].real(), sum[ii][jj][iii][jjj].imag());
                        }
                    }
                }
            }
        }
    }


    // Call DPC++ HERK.
    oneapi::mkl::blas::row_major::herk(main_queue, upper_lower, trans, n, k,
                                       alpha, A.data(), lda, beta, C_ref.data(), ldc,
                                       dependencies).wait();

    try {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                throw oneapi::mkl::unimplemented{"Unkown", "Unkown"};
                break;
            case oneapi::mkl::layout::row_major:
                done = t2sp::blas::row_major::herk(fpga_queue, upper_lower, trans, n, k,
                                                   alpha, A.data(), lda, beta, C.data(), ldc,
                                                   dependencies);
                break;
            default: break;
        }
        done.wait();
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during HERK:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of HERK:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_matrix(C, C_ref, layout, n, n, ldc, 10 * std::max(n, k), std::cout);

    return (int)good;
}

class HerkUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(HerkUsmTests, ComplexSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::nontrans, 72, 28, 101, 103, alpha, beta)));
#elif defined(T2SP_TEST_1)
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans, 72, 28, 101, 103, alpha, beta)));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::conjtrans, 72, 28, 101, 103, alpha, beta)));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::conjtrans, 72, 28, 101, 103, alpha, beta)));
#endif
}
TEST_P(HerkUsmTests, ComplexDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::nontrans, 72, 28, 101, 103, alpha, beta)));
#elif defined(T2SP_TEST_1)
//    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
//        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
//        oneapi::mkl::transpose::nontrans, 72, 28, 101, 103, alpha, beta)));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::conjtrans, 72, 28, 101, 103, alpha, beta)));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::conjtrans, 72, 28, 101, 103, alpha, beta)));
#endif
}

INSTANTIATE_TEST_SUITE_P(HerkUsmTestSuite, HerkUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
