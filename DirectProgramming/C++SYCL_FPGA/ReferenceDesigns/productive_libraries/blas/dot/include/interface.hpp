#pragma once
#include <iostream>
#include <sstream>
#include <type_traits>
#include <tuple>

#include "HalideBuffer.h"

#ifdef TINY
#include "sdot.tiny.sycl.h"
#include "ddot.tiny.sycl.h"
#else
#include "sdot.large.sycl.h"
#include "ddot.large.sycl.h"
#endif

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
    // but since We need the vector length to be a multiple of KKK,
    // this requires the macros KKK.
    template <typename T>
    constexpr auto get_dimensions() {
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
        } else {
            return 4;
        }
#endif
    }
}

#define t2sp_assert(c) \
    (c) ? (void)0 : ::t2sp::detail::Voidifier{} & ::t2sp::detail::ErrorReport(__FILE__, __LINE__, #c).ref()

template <typename T>
size_t dot(const int N, const T *x, const int incx, const T *y, const int incy, T *result) {
    using Halide::Runtime::Buffer;

    constexpr auto KKK = detail::get_dimensions<T>();

    t2sp_assert(N % KKK == 0) << "Due to the limitations of our current implementation, N must be a multiple of " << KKK << ", but N = " << N;

    halide_dimension_t x_dim[]{{0, N, std::abs(incx)}, {0, 1, 1}};
    halide_dimension_t y_dim[]{{0, N, std::abs(incy)}, {0, 1, 1}};
    halide_dimension_t result_dim[]{{0, 1, 1}};

    Buffer<T> x_buffer{const_cast<T *>(x), 2, x_dim};
    Buffer<T> y_buffer{const_cast<T *>(y), 2, y_dim};
    Buffer<T> result_buffer{result, 1, result_dim};

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

    size_t exec_time = 0;
    if constexpr (std::is_same_v<T, float>) {
        exec_time = sdot::sdot(device_selector, std::abs(incx), std::abs(incy), x_buffer, y_buffer, result_buffer);
    } else if constexpr (std::is_same_v<T, double>) {
        exec_time = ddot::ddot(device_selector, std::abs(incx), std::abs(incy), x_buffer, y_buffer, result_buffer);
    } else {
        std::cerr << "This type is not currently supported in DOT\n";
    }
    return exec_time;
}
}

