// To compile this file, pass in a macro for the compute (T2SP_S/D/C/ZMATMUL), the size of the systolic array (TINY or LARGE), and the hardware(A10 or S10).
// And pass in a macro FPGA_EMULATOR if to use the emulator instead of FPGA hardware.


#pragma once
#include <complex>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <tuple>

#include "HalideBuffer.h"
#include "./parameters.h"

// Declarations
namespace t2sp {
// The BLAS style API, which calls the Halide style API below internally.
size_t blas_matmul(bool transa, bool transb, int64_t m, int64_t n, int64_t k, CONST_TYPE alpha, const CONST_TYPE *a, int64_t lda,  const CONST_TYPE *b, int64_t ldb, CONST_TYPE beta, CONST_TYPE *c, int64_t ldc);

// The Halide style API. Its implementation has been automatically generated as a static library. We just need to link this file with that library.
extern uint64_t matmul(device_selector_t device_selector_v, bool transa, bool transb, CONST_TYPE alpha, CONST_TYPE beta,
                       struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer, struct halide_buffer_t *Output_buffer);
}

// Implementation of the BLAS style API
namespace t2sp {
namespace detail {
    struct ErrorReport {
        std::ostringstream msg;
        std::string_view file, condition;
        int line;
        ErrorReport(std::string_view file, int line, std::string_view condition)
            : msg{}, file{file}, line{line}, condition{condition} {}
        ErrorReport &operator<<(const CONST_TYPE &x) {
            msg << x;
            return *this;
        }
        ErrorReport &ref() { return *this; }
        ~ErrorReport() noexcept(false) {
            if (!msg.str().empty() && msg.str().back() != '\n')
                msg << '\n';
            std::cerr << "Assertion (" << condition << ") failed at "
                      << file << ":" << line << "\n\t"
                      << msg.str();
            std::abort();
        }
    };
    struct Voidifier {
        void operator &(ErrorReport &) const {}
    };
}

#define t2sp_assert(c) \
    (c) ? (void)0 : ::t2sp::detail::Voidifier{} & ::t2sp::detail::ErrorReport(__FILE__, __LINE__, #c).ref()

size_t blas_matmul(bool transa, bool transb, int m, int n, int k, CONST_TYPE alpha, const CONST_TYPE *a, int lda,  const CONST_TYPE *b, int ldb, CONST_TYPE beta, CONST_TYPE *c, int ldc) {
    // Check parameters according to BLAS interface conventions
    t2sp_assert(m >= 0 && n >= 0 && k >= 0) << "m = " << m << ", n = " << n << ", k = " << k;
    t2sp_assert(a && b && c) << "a = " << (const void *)a << ", b = " << (const void *)b << ", c = " << (const void *)c;
    t2sp_assert(!transa && lda >= std::max(k, 1) || transa && lda >= std::max(m, 1))
        << std::boolalpha << "transa = " << transa << ", lda = " << lda << ", m = " << m << ", k = " << k;
    t2sp_assert(!transb && ldb >= std::max(n, 1) || transb && ldb >= std::max(k, 1))
        << std::boolalpha << "transb = " << transb << ", ldb = " << ldb << ", n = " << n << ", k = " << k;
    t2sp_assert(ldc >= std::max(n ,1))
        << "ldc = " << ldc << ", n = " << n;
               
    t2sp_assert(n % JJJ == 0) << "Due to the limitations of our current implementation, n must be a multiple of " << JJJ
                              << "(the vectorized dimension for the output matrix), but n = " << n;
    t2sp_assert(k % KKK == 0) << "Due to the limitations of our current implementation, k must be a multiple of " << KKK
                              << "(the vectorized dimension for the input matrices), but k = " << k;

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_a[]{{0, k, 1}, {0, m, lda}};
    Buffer<CONST_TYPE> buffer_a{const_cast<CONST_TYPE *>(a), 2, dim_a};
    halide_dimension_t dim_b[]{{0, n, 1}, {0, k, ldb}};
    Buffer<CONST_TYPE> buffer_b{const_cast<CONST_TYPE *>(b), 2, dim_b};
    halide_dimension_t dim_c[]{{0, n, 1}, {0, m, ldc}};
    Buffer<CONST_TYPE> buffer_c{c, 2, dim_c};
    Buffer<CONST_TYPE> buffer_out{JJJ, JJ, II, III, (n + (JJJ * JJ - 1)) / (JJJ * JJ), (m + (III * II - 1)) / (III * II)};

#if defined(SYCL_LANGUAGE_VERSION) && SYCL_LANGUAGE_VERSION >= 202001
#if defined(FPGA_EMULATOR)
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
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector{};
#else
    sycl::ext::intel::fpga_selector device_selector{};
#endif
#endif

    ret = t2sp::matmul(device_selector, transa, transb, alpha, beta, buffer_a, buffer_b, buffer_c, buffer_out);

    // Collect the results and remove the extraneous data that are beyond the dimensions of the output matrix.
    for (int i = 0; i < (m + (III * II - 1)) / (III * II); i++) {
        for (int j = 0; j < (n + (JJJ * JJ - 1)) / (JJJ * JJ); j++) {
            for (int ii = 0; ii < II; ii++) {
                for (int jj = 0; jj < JJ; jj++) {
                    for (int iii = 0; iii < III; iii++) {
                        for (int jjj = 0; jjj < JJJ; jjj++) {
                            int total_i = iii + III * ii + III * II * i;
                            int total_j = jjj + JJJ * jj + JJJ * JJ * j;
                            if (total_i < m && total_j < n) {
                                c[total_j + total_i * ldc] = buffer_out(jjj, jj, ii, iii, j, i);
                            }
                        }
                    }
                }
            }
        } 
    }
    return ret;
}
}
