#pragma once

#include "HalideBuffer.h"

namespace t2sp::blas::row_major {
extern uint64_t smatmul(device_selector_t device_selector_v, bool FromSymmetricPosA, bool FromSymmetricPosB, bool FromSymmetricPosC,
                       bool ConjugateA, bool ConjugateB, bool ConjugateC, bool HalfSpace, float alpha, float beta,
                       struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer,
                       struct halide_buffer_t *Output_buffer);

extern uint64_t dmatmul(device_selector_t device_selector_v, bool FromSymmetricPosA, bool FromSymmetricPosB, bool FromSymmetricPosC,
                       bool ConjugateA, bool ConjugateB, bool ConjugateC, bool HalfSpace, double alpha, double beta,
                       struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer,
                       struct halide_buffer_t *Output_buffer);

extern uint64_t cmatmul(device_selector_t device_selector_v, bool FromSymmetricPosA, bool FromSymmetricPosB, bool FromSymmetricPosC,
                       bool ConjugateA, bool ConjugateB, bool ConjugateC, bool HalfSpace, std::complex<float> alpha, std::complex<float> beta,
                       struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer,
                       struct halide_buffer_t *Output_buffer);

extern uint64_t zmatmul(device_selector_t device_selector_v, bool FromSymmetricPosA, bool FromSymmetricPosB, bool FromSymmetricPosC,
                       bool ConjugateA, bool ConjugateB, bool ConjugateC, bool HalfSpace, std::complex<double> alpha, std::complex<double> beta,
                       struct halide_buffer_t *A_buffer, struct halide_buffer_t *B_buffer, struct halide_buffer_t *C_buffer,
                       struct halide_buffer_t *Output_buffer);
}
