#pragma once

#include <sycl/sycl.hpp>
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
namespace sdot {
extern sycl::event sdot(sycl::queue &, bool, halide_buffer_t *, int, halide_buffer_t *, int, halide_buffer_t *);
}

namespace ddot {
extern sycl::event ddot(sycl::queue &, bool, halide_buffer_t *, int, halide_buffer_t *, int, halide_buffer_t *);
}

namespace cdot {
extern sycl::event cdot(sycl::queue &, bool, halide_buffer_t *, int, halide_buffer_t *, int, halide_buffer_t *);
}

namespace cdot {
extern sycl::event zdot(sycl::queue &, bool, halide_buffer_t *, int, halide_buffer_t *, int, halide_buffer_t *);
}

// Query of the parameters of the systolic array (KKK) based on types
template <typename T>
constexpr auto get_systolic_array_dimensions() {
    _halide_user_assert((std::is_same_v<float, T>) ||
                        (std::is_same_v<double, T>) ||
                        (std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";
#ifdef TINY
    return 4;
#else
#ifdef S10
    constexpr bool run_on_s10 = true;
#else
    constexpr bool run_on_s10 = false;
#endif
    if constexpr (std::is_same_v<T, float>) {
        return run_on_s10 ? 32 : 16; 
    } else if constexpr (std::is_same_v<T, double>) {
        return run_on_s10 ? 16 : 8; 
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return run_on_s10 ? 16 : 8; 
    } else {
        return run_on_s10 ? 8 : 4; 
    }
#endif
}

} // namespace t2sp::blas::row_major
