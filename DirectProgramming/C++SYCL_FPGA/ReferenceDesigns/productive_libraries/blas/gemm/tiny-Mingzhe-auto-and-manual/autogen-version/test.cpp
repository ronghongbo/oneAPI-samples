#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <type_traits>
#include <vector>

#include "interface.hpp"
#include "oneapi/mkl.hpp"

template <typename T, int align>
struct allocator_helper {
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef void* void_pointer;
    typedef const void* const_void_pointer;
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    template <typename U>
    struct rebind {
        typedef allocator_helper<U, align> other;
    };

    allocator_helper() noexcept {}
    template <typename U, int align2>
    allocator_helper(allocator_helper<U, align2>& other) noexcept {}
    template <typename U, int align2>
    allocator_helper(allocator_helper<U, align2>&& other) noexcept {}

    T* allocate(size_t n) {
#ifdef _WIN64
        void* mem = ::_aligned_alloc(n * sizeof(T), align);
#else
        void *mem = ::aligned_alloc(align, n * sizeof(T));
#endif
        if (!mem)
            throw std::bad_alloc();

        return static_cast<T*>(mem);
    }

    void deallocate(T* p, size_t n) noexcept {
#ifdef _WIN64
        ::_aligned_free(p);
#else
        ::free(p);
#endif
    }

    constexpr size_t max_size() const noexcept {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }

    template <typename U, int align2>
    constexpr bool operator==(const allocator_helper<U, align2>) const noexcept {
        return true;
    }
    template <typename U, int align2>
    constexpr bool operator!=(const allocator_helper<U, align2>) const noexcept {
        return false;
    }

    typedef std::true_type is_always_equal;
};


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
void rand_matrix(vec &M, oneapi::mkl::transpose trans, int m, int n, int ld) {
    using fp = typename vec::value_type;

    M.resize(trans == oneapi::mkl::transpose::nontrans ? m * ld : n * ld);

    if (trans != oneapi::mkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    } else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
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

template <typename acc1, typename acc2>
bool check_equal_matrix(acc1 &M, acc2 &M_ref, int m, int n, int ld,
                        int error_mag, std::ostream &out) {
    bool good = true;
    int idx, count = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            idx = j + i * ld;
            if (!check_equal(M[idx], M_ref[idx], error_mag)) {
                out << "Difference in entry (" << i << ',' << j << "): t2sp " << M[idx]
                    << " vs. Reference (oneMKL) " << M_ref[idx] << std::endl;
                good = false;
                count++;
                if (count > 20)
                    return good;
            }
        }
    }

    return good;
}

template <typename Ta, typename Tc>
bool test_internal(oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
         int m, int n, int k, int lda, int ldb, int ldc, Tc alpha, Tc beta) {
    using std::vector;
    vector<Ta, allocator_helper<Ta, 64>> A, B;
    vector<Ta, allocator_helper<Ta, 64>> C, C_ref;

    rand_matrix(A, transa, m, k, lda);
    rand_matrix(B, transb, k, n, ldb);
    rand_matrix(C, oneapi::mkl::transpose::nontrans, m, n, ldc);
    
    C_ref = C;

    t2sp::gemm(transa != oneapi::mkl::transpose::nontrans,
               transb != oneapi::mkl::transpose::nontrans,
               m, n, k, alpha, A.data(), lda, B.data(), ldb,
               beta, C.data(), ldc);

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

    sycl::buffer<Ta, 1> A_buffer(A.data(), sycl::range<1>(A.size()));
    sycl::buffer<Ta, 1> B_buffer(B.data(), sycl::range<1>(B.size()));
    sycl::buffer<Tc, 1> C_ref_buffer(C_ref.data(), sycl::range<1>(C_ref.size()));

    oneapi::mkl::blas::row_major::gemm(main_queue, transa, transb, m, n, k, alpha,
                                       A_buffer, lda, B_buffer, ldb, beta, C_ref_buffer,
                                       ldc);

    auto C_ref_accessor = C_ref_buffer.template get_host_access(sycl::read_only);
    return check_equal_matrix(C, C_ref_accessor, m, n, ldc, 10 * k, std::cout);
}

template <typename T>
void test(oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
         int m, int n, int k, int lda, int ldb, int ldc, T alpha, T beta) {
    const char *name = nullptr;
    if constexpr (std::is_same_v<float, T>)
        name = "sgemm";
    else if constexpr (std::is_same_v<double, T>)
        name = "dgemm";
    else if constexpr (std::is_same_v<std::complex<float>, T>)
        name = "cgemm";
    else if constexpr (std::is_same_v<std::complex<double>, T>)
        name = "zgemm";
    else
        name = "unsupported data type";
    std::cout << "\ntest for " << name << ":\n\t"
              << "op(A) = " << (transa == oneapi::mkl::transpose::trans ? "A^T" : "  A") << ", m = "
              << std::setw(3) << m << ", k = " << std::setw(3) << k << ", lda = " << std::setw(3) << lda << "\n\t"
              << "op(B) = " << (transb == oneapi::mkl::transpose::trans ? "B^T" : "  B") << ", k = "
              << std::setw(3) << k << ", n = " << std::setw(3) << n << ", ldb = " << std::setw(3) << ldb << "\n\t"
              << "   C  =   C" << ", m = "
              << std::setw(3) << m << ", n = " << std::setw(3) << n << ", ldc = " << std::setw(3) << ldc << "\n\t"
              << "alpha = " << alpha << "\n\t"
              << " beta = " << beta << "\n";
    std::cout << (test_internal<T, T>(transa, transb, m, n, k, lda, ldb, ldc, alpha, beta)
            ? "\x1b[1;32mtest succeed\x1b[0m\n" : "\x1b[1;31mtest failed!!!\x1b[0m\n");
}

void test_all(oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
              int m, int n, int k, int lda, int ldb, int ldc) {
    test<float>(transa, transb, m, n, k, lda, ldb, ldc, 2.0f, 3.0f);
    test<double>(transa, transb, m, n, k, lda, ldb, ldc, 2.0, 3.0);
    test<std::complex<float>>(transa, transb, m, n, k, lda, ldb, ldc, {2.0f, -0.5f}, {3.0f, -1.5f});
    test<std::complex<double>>(transa, transb, m, n, k, lda, ldb, ldc, {2.0, -0.5}, {3.0, -1.5});
}

// Due to issues with the current autorun kernel implementation, we can only test one at a time
int main() {
#ifdef T2SP_TEST_0
    test_all(oneapi::mkl::transpose::nontrans,
             oneapi::mkl::transpose::nontrans,  3,  8, 12, 103, 105, 106);
#elif defined(T2SP_TEST_1)
    test_all(oneapi::mkl::transpose::nontrans,
             oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106);
#elif defined(T2SP_TEST_2)
    test_all(oneapi::mkl::transpose::nontrans,
                oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106);
#elif defined(T2SP_TEST_3)
    test_all(   oneapi::mkl::transpose::trans,
             oneapi::mkl::transpose::nontrans, 79, 84, 92, 103, 105, 106);
#elif defined(T2SP_TEST_4)
    test_all(   oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::trans, 79, 84, 92, 103, 105, 106);
#endif
}
