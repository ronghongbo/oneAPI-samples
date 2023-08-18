#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// The dot API to invoke
#include "./api.hpp"

// Useful routines from the OneMKL unit tests
#include "allocator_helper.hpp"
#include "oneapi/mkl.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include "exception_handler.hpp"

using namespace std;

template <typename T_res, typename T>
void test(int N, int incx, int incy) {
    vector<T, allocator_helper<T, 64>> x, y;
    T_res res{};
    rand_vector(x, N, incx);
    rand_vector(y, N, incy);

    sycl::queue q_device(sycl::ext::intel::fpga_selector_v, fpga_tools::exception_handler, sycl::property::queue::enable_profiling());

    auto done = t2sp::blas::row_major::dot(q_device, N, x.data(), incx, y.data(),
                                           incy, &res);
    done.wait();

    // Get time in ns
    uint64_t start = done.template get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end   = done.template get_profiling_info<sycl::info::event_profiling::command_end>();
    uint64_t exec_time = end - start;
    std::cout << "Execution time in nanoseconds = " << exec_time << "\n";

    double number_ops = 2.0 * N;
    std::cout << "GFLOPs: " << number_ops / exec_time << "\n";
    std::cout << "Size of vector x: " << N << "\n";
    std::cout << "Size of vector y: " << N << "\n";
}

int main() {
#if defined(T2SP_SDOT)
    using vectors_data_type = float;
    using res_type = float;
#elif defined(T2SP_DDOT)
    using vectors_data_type = double;
    using res_type = double;
#elif defined(T2SP_DSDOT)
    using vectors_data_type = float;
    using res_type = double;
#else
#error No test and result type specified
#endif
    const auto KKK = t2sp::blas::row_major::get_systolic_array_dimensions<vectors_data_type>();
    int64_t n = KKK * 4096 * 4096;
    test<res_type, vectors_data_type>(n, 1, 1);
}
