#pragma once

#include <sycl/sycl.hpp>
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
extern sycl::event smatmul(sycl::queue &q_device, bool FromSymmetricPosA, bool FromSymmetricPosB, bool FromSymmetricPosC,
                           bool ConjugateA, bool ConjugateB, bool ConjugateC, bool HalfSpace, float alpha, float beta,
                           struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer,
                           struct halide_buffer_t *Output_buffer);

extern sycl::event dmatmul(sycl::queue &q_device, bool FromSymmetricPosA, bool FromSymmetricPosB, bool FromSymmetricPosC,
                           bool ConjugateA, bool ConjugateB, bool ConjugateC, bool HalfSpace, double alpha, double beta,
                           struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer,
                           struct halide_buffer_t *Output_buffer);

extern sycl::event cmatmul(sycl::queue &q_device, bool FromSymmetricPosA, bool FromSymmetricPosB, bool FromSymmetricPosC,
                           bool ConjugateA, bool ConjugateB, bool ConjugateC, bool HalfSpace, std::complex<float> alpha, std::complex<float> beta,
                           struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer,
                           struct halide_buffer_t *Output_buffer);

extern sycl::event zmatmul(sycl::queue &q_device, bool FromSymmetricPosA, bool FromSymmetricPosB, bool FromSymmetricPosC,
                           bool ConjugateA, bool ConjugateB, bool ConjugateC, bool HalfSpace, std::complex<double> alpha, std::complex<double> beta,
                           struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer,
                           struct halide_buffer_t *Output_buffer);



// Query of the parameters of the systolic array (KKK, JJJ, III, JJ, II, KK) based on types
template <typename T>
constexpr auto get_systolic_array_dimensions() {
    _halide_user_assert((std::is_same_v<float, T>) ||
                        (std::is_same_v<double, T>) ||
                        (std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";
#ifdef TINY
    return std::tuple{4, 4, 4, 4, 4, 4};
#else
#ifdef S10
    constexpr bool run_on_s10 = true;
#else
    constexpr bool run_on_s10 = false;
#endif
    if constexpr (std::is_same_v<T, float>) {
        return run_on_s10 ? std::tuple{16, 16, 14, 32, 32, 32} : std::tuple{16, 8, 10, 32, 32, 32};
    } else if constexpr (std::is_same_v<T, double>) {
        return run_on_s10 ? std::tuple{8, 4, 8, 32, 32, 32} : std::tuple{8, 4, 8, 32, 32, 32};
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return run_on_s10 ? std::tuple{16, 16, 14, 32, 32, 32} : std::tuple{8, 4, 10, 32, 32, 32};
    } else {
        return run_on_s10 ? std::tuple{4, 6, 4, 32, 32, 32} : std::tuple{4, 4, 4, 32, 32, 32};
    }
#endif
}

} // namespace t2sp::blas::row_major
