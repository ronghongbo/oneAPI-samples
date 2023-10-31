#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "complex_helper.hpp"
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
namespace smvmul {
extern sycl::event smvmul(sycl::queue &, bool, float, halide_buffer_t *, int, halide_buffer_t *, float, int, halide_buffer_t *);
}

namespace dmvmul {
extern sycl::event dmvmul(sycl::queue &, bool, double, halide_buffer_t *, int, halide_buffer_t *, double, int, halide_buffer_t *);
}

namespace cmvmul {
extern sycl::event cmvmul(sycl::queue &, bool, complexf, halide_buffer_t *, int, halide_buffer_t *, complexf, int, halide_buffer_t *);
}

namespace zmvmul {
extern sycl::event zmvmul(sycl::queue &, bool, zomplexf, halide_buffer_t *, int, halide_buffer_t *, zomplexf, int, halide_buffer_t *);
}

// Query of the parameters of the systolic array (KKK) based on types
template <typename T>
constexpr auto get_systolic_array_dimensions() {
    _halide_user_assert((std::is_same_v<float, T>) ||
                        (std::is_same_v<double, T>) ||
                        (std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";
#ifdef TINY
    return std::tuple{4, 4, 4};
#else
#ifdef S10
    constexpr bool run_on_s10 = true;
#else
    constexpr bool run_on_s10 = false;
#endif
    return run_on_s10 ? std::tuple{64, 32, 32} : std::tuple{64, 32, 32};
#endif
}

} // namespace t2sp::blas::row_major
