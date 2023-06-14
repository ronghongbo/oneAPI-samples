#pragma once

#include <sycl/sycl.hpp>
#include "HalideBuffer.h"

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
}
