#pragma once
#include <complex>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <tuple>

#include "HalideBuffer.h"
#include "sgemm.sycl.h"
#include "dgemm.sycl.h"
#include "cgemm.sycl.h"
#include "zgemm.sycl.h"

namespace t2sp {

namespace detail {
    struct ErrorReport {
        std::ostringstream msg;
        std::string_view file, condition;
        int line;
        ErrorReport(std::string_view file, int line, std::string_view condition)
            : msg{}, file{file}, line{line}, condition{condition} {}
        template <typename T>
        ErrorReport &operator<<(const T &x) {
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
    // Our interface shouldn't depend on these macro definitions,
    // but since we're going to strip out extra zeros next,
    // this requires the macros II/JJ/III/JJJ.
    // When we improve our implementation in the future,
    // we can remove this part.
    template <typename T>
    constexpr auto get_dimensions() {
#ifdef TINY
        return std::tuple{4, 4, 4, 4, 4};
#else
#ifdef S10
        constexpr bool run_on_s10 = true;
#else
        constexpr bool run_on_s10 = false;
#endif
        if constexpr (std::is_same_v<T, float>) {
            return run_on_s10 ? std::tuple{32, 32, 14, 16, 16} : std::tuple{32, 32, 10, 8, 16};
        } else if constexpr (std::is_same_v<T, double>) {
            return run_on_s10 ? std::tuple{32, 32, 8, 4, 8} : std::tuple{32, 32, 8, 4, 8};
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            return run_on_s10 ? std::tuple{32, 32, 14, 16, 16} : std::tuple{32, 32, 10, 4, 8};
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            return run_on_s10 ? std::tuple{32, 32, 4, 6, 4} : std::tuple{32, 32, 4, 4, 4};
        } else {
            return std::tuple{4, 4, 4, 4, 4};
        }
#endif
    }
}

#define t2sp_assert(c) \
    (c) ? (void)0 : ::t2sp::detail::Voidifier{} & ::t2sp::detail::ErrorReport(__FILE__, __LINE__, #c).ref()

template <typename T>
size_t gemm(bool transa, bool transb, int m, int n, int k, T alpha, const T *a, int lda,  const T *b, int ldb, T beta, T *c, int ldc) {
    // Check parameters according to BLAS interface conventions
    t2sp_assert(m >= 0 && n >= 0 && k >= 0) << "m = " << m << ", n = " << n << ", k = " << k;
    t2sp_assert(a && b && c) << "a = " << (const void *)a << ", b = " << (const void *)b << ", c = " << (const void *)c;
    t2sp_assert(!transa && lda >= std::max(k, 1) || transa && lda >= std::max(m, 1))
        << std::boolalpha << "transa = " << transa << ", lda = " << lda << ", m = " << m << ", k = " << k;
    t2sp_assert(!transb && ldb >= std::max(n, 1) || transb && ldb >= std::max(k, 1))
        << std::boolalpha << "transb = " << transb << ", ldb = " << ldb << ", n = " << n << ", k = " << k;
    t2sp_assert(ldc >= std::max(n ,1))
        << "ldc = " << ldc << ", n = " << n;
               
    auto [II, JJ, III, JJJ, KKK] = detail::get_dimensions<T>();

    t2sp_assert(n % JJJ == 0) << "Due to the limitations of our current implementation, n must be a multiple of " << JJJ << ", but n = " << n;
    t2sp_assert(k % KKK == 0) << "Due to the limitations of our current implementation, k must be a multiple of " << KKK << ", but k = " << k;

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_a[]{{0, k, 1}, {0, m, lda}};
    Buffer<T> buffer_a{const_cast<T *>(a), 2, dim_a};
    halide_dimension_t dim_b[]{{0, n, 1}, {0, k, ldb}};
    Buffer<T> buffer_b{const_cast<T *>(b), 2, dim_b};
    halide_dimension_t dim_c[]{{0, n, 1}, {0, m, ldc}};
    Buffer<T> buffer_c{c, 2, dim_c};
    Buffer<T> buffer_out{JJJ, JJ, II, III, (n + (JJJ * JJ - 1)) / (JJJ * JJ), (m + (III * II - 1)) / (III * II)};

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

    size_t ret = 0;
    if constexpr (std::is_same_v<T, float>) {
        ret = sgemm::sgemm(device_selector, transa, transb, alpha, beta, buffer_a, buffer_b, buffer_c, buffer_out);
    } else if constexpr (std::is_same_v<T, double>) {
        ret = dgemm::dgemm(device_selector, transa, transb, alpha, beta, buffer_a, buffer_b, buffer_c, buffer_out);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        ret = cgemm::cgemm(device_selector, transa, transb, alpha, beta, buffer_a, buffer_b, buffer_c, buffer_out);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        ret = zgemm::zgemm(device_selector, transa, transb, alpha, beta, buffer_a, buffer_b, buffer_c, buffer_out);
    } else {
        std::cerr << "This type is not currently supported in GEMM\n";
        return 0;
    }

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
