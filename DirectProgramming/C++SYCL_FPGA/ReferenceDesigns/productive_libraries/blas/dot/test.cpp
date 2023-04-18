#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <type_traits>
#include <vector>

#include "interface.hpp"
#include "oneapi/mkl.hpp"

// Random initialization.
template <typename fp>
fp rand_scalar() {
    if constexpr (std::is_same_v<fp, std::complex<float>> ||
                  std::is_same_v<fp, std::complex<double>>) {
        return fp(rand_scalar<typename fp::value_type>(), rand_scalar<typename fp::value_type>());
    } else {
        return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
    }
}

template <typename vec>
void rand_vector(vec &v, int n, int inc) {
    using fp = typename vec::value_type;
    int abs_inc = std::abs(inc);

    v.resize(n * abs_inc);

    for (int i = 0; i < n; i++)
        v[i * abs_inc] = rand_scalar<fp>();
}

// Correctness checking.
template <typename T>
constexpr auto helper() noexcept {
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        return 2 * std::numeric_limits<typename T::value_type>::epsilon();
    } else {
        return std::numeric_limits<T>::epsilon();
    }
}

template <typename fp>
typename std::enable_if<!std::is_integral<fp>::value, bool>::type check_equal(fp x, fp x_ref,
                                                                              int error_mag) {
    auto bound = error_mag * helper<fp>();
    bool ok = false;

    auto aerr = std::abs(x - x_ref);
    auto rerr = aerr / std::abs(x_ref);
    ok = (rerr <= bound) || (aerr <= bound);
    if (!ok)
        std::cout << "relative error = " << rerr << " absolute error = " << aerr
                  << " limit = " << bound << std::endl;
    return ok;
}

template <typename fp>
bool check_equal(fp x, fp x_ref, int error_mag, std::ostream &out) {
    bool good = check_equal(x, x_ref, error_mag);

    if (!good) {
        out << "Difference in result: T2SP " << x << " vs. oneMKL " << x_ref << std::endl;
    }
    return good;
}

template <typename fp, typename fp_res>
bool test_internal(int N, int incx, int incy) {
    using std::vector;
    vector<fp> x, y;
    fp_res result = fp_res(-1), result_ref = fp_res(-1);

    rand_vector(x, N, incx);
    rand_vector(y, N, incy);
    
    t2sp::dot(N, x.data(), incx, y.data(), incy, &result);

    // Call oneMKL GEMM as Reference.
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
                          << e.what() << std::endl;
            }
        }
    };
    sycl::queue main_queue(sycl::cpu_selector_v, exception_handler);

    sycl::buffer<fp, 1> x_buffer(x.data(), sycl::range<1>(x.size()));
    sycl::buffer<fp, 1> y_buffer(y.data(), sycl::range<1>(y.size()));
    sycl::buffer<fp_res, 1> result_ref_buffer(&result_ref, sycl::range<1>(1));

    oneapi::mkl::blas::row_major::dot(main_queue, N, x_buffer, incx, y_buffer, incy, result_ref_buffer);

    auto result_ref_accessor = result_ref_buffer.template get_host_access(sycl::read_only);
    return check_equal(result, result_ref_accessor[0], N, std::cout);
}

template <typename T>
void test(int N, int incx, int incy) {
    const char *name = nullptr;
    if constexpr (std::is_same_v<float, T>)
        name = "sdot";
    else if constexpr (std::is_same_v<double, T>)
        name = "ddot";
    else
        name = "unsupported data type";
    std::cout << "\ntest for " << name << ":\n\t"
              << "x: N = " << std::setw(4) << N << ", incx = " << std::setw(4) << incx << "\n\t"
              << "y: N = " << std::setw(4) << N << ", incy = " << std::setw(4) << incy << "\n";
              
    std::cout << (test_internal<T, T>(N, incx, incy)
            ? "\x1b[1;32mtest succeed\x1b[0m\n" : "\x1b[1;31mtest failed!!!\x1b[0m\n");
}

void test_all(int N, int incx, int incy) {
    test<float>(N, incx, incy);
    test<double>(N, incx, incy);
}

int main() {
    test_all(1360, 1, 1);
    test_all(1360, 2, 3);
    test_all(1360, -3, -2);
}
