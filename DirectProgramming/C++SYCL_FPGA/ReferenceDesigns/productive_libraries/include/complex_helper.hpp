#pragma once
#include <utility>
#include <complex>
#include <array>
#include <sycl/sycl.hpp>

using complexf = std::complex<float>;
using complexd = std::complex<double>;

namespace t2sp {
namespace detail {

template <typename T, size_t N> struct storage_type {};
template <size_t N>
struct storage_type<std::complex<float>, N> {
    using type = sycl::vec<float, N * 2>;
}; 
template <size_t N>
struct storage_type<std::complex<double>, N> {
    using type = sycl::vec<double, N * 2>;
};

template <typename T> struct unwrap {};
template <typename T>
struct unwrap<std::complex<T>> {
    using type = T;
};

template <typename T, size_t N>
class vec {
    typename storage_type<T, N>::type _data;
    template <size_t... Is>
    void make_vec(const std::array<T, N> &arr, std::index_sequence<Is...>) {
        ((_data[2 * Is] = arr[Is].real()), ...);
        ((_data[2 * Is + 1] = arr[Is].imag()), ...);
    }
  public:
    vec() = default;
    vec(const vec &) = default;
    vec(vec &&) = default;
    vec &operator=(const vec &) = default;
    vec &operator=(vec &&) = default;
    vec(const T &arg) {
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            _data[2 * i] = arg.real();
            _data[2 * i + 1] = arg.imag();
        }
    }
    template <typename ...Args>
    vec(const Args &...args) {
        make_vec(std::array{args...}, std::make_index_sequence<N>());
    }
    vec &operator=(const T& arg) {
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            _data[2 * i] = arg.real();
            _data[2 * i + 1] = arg.imag();
        }
        return *this;
    }
    T &operator[](size_t n) {
        return reinterpret_cast<T*>(&_data)[n];
    }
    const T &operator[](size_t n) const {
        return reinterpret_cast<const T*>(&_data)[n];
    }
    constexpr static size_t size() noexcept {
        return N;
    }
    vec &operator+=(const vec &rhs) {
        _data += rhs._data;
        return *this;
    }
    vec &operator+=(const T &rhs) {
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            _data[2 * i] += rhs.real();
            _data[2 * i + 1] += rhs.imag();
        }
        return *this;
    }
    vec &operator-=(const vec &rhs) {
        _data -= rhs._data;
        return *this;
    }
    vec &operator-=(const T &rhs) {
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            _data[2 * i] -= rhs.real();
            _data[2 * i + 1] -= rhs.imag();
        }
        return *this;
    }
    vec &operator*=(const vec &rhs) {
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            _data[2 * i] = _data[2 * i] * rhs._data[2 * i] - _data[2 * i + 1] * rhs._data[2 * i + 1];
            _data[2 * i + 1] = _data[2 * i] * rhs._data[2 * i + 1] + _data[2 * i + 1] * rhs._data[2 * i];
        }
        return *this;
    }
    vec &operator*=(const T &arg) {
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            _data[2 * i] = _data[2 * i] * arg.real() - _data[2 * i + 1] * arg.imag();
            _data[2 * i + 1] = _data[2 * i] * arg.imag() + _data[2 * i + 1] * arg.real();
        }
        return *this;
        return *this;
    }
    vec conj() const {
        vec ret{};
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret[2 * i] = _data[2 * i];
            ret[2 * i + 1] = -_data[2 * i + 1];
        }
        return ret;
    }
    friend vec operator+(const vec &arg) {
        return arg;
    }
    friend vec operator-(const vec &arg) {
        vec ret{};
        ret._data = -arg._data;
        return ret;
    }
    friend vec operator+(const vec &lhs, const vec &rhs) {
        vec ret{};
        ret._data = lhs._data + rhs._data;
        return ret;
    }
    friend vec operator+(const vec &lhs, const T &rhs) {
        vec ret{};
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret._data[2 * i] = lhs._data[2 * i] + rhs.real();
            ret._data[2 * i + 1] = lhs._data[2 * i + 1] + rhs.imag();
        }
        return ret;
    }
    friend vec operator+(const T &lhs, const vec &rhs) {
        vec ret{};
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret._data[2 * i] = lhs.real() + rhs._data[2 * i];
            ret._data[2 * i + 1] = lhs.imag() + rhs._data[2 * i + 1];
        }
        return ret;
    }
    friend vec operator-(const vec &lhs, const vec &rhs) {
        vec ret{};
        ret._data = lhs._data - rhs._data;
        return ret;
    }
    friend vec operator-(const vec &lhs, const T &rhs) {
        vec ret{};
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret._data[2 * i] = lhs._data[2 * i] - rhs.real();
            ret._data[2 * i + 1] = lhs._data[2 * i + 1] - rhs.imag();
        }
        return ret;
    }
    friend vec operator-(const T &lhs, const vec &rhs) {
        vec ret{};
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret._data[2 * i] = lhs.real() - rhs._data[2 * i];
            ret._data[2 * i + 1] = lhs.imag() - rhs._data[2 * i + 1];
        }
        return ret;
    }
    friend vec operator*(const vec &lhs, const vec &rhs) {
        vec ret{};
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret._data[2 * i] = lhs._data[2 * i] * rhs._data[2 * i] - lhs._data[2 * i + 1] * rhs._data[2 * i + 1];
            ret._data[2 * i + 1] = lhs._data[2 * i] * rhs._data[2 * i + 1] + lhs._data[2 * i + 1] * rhs._data[2 * i];
        }
        return ret;
    }
    friend vec operator*(const vec &lhs, const T &rhs) {
        vec ret{};
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret._data[2 * i] = lhs._data[2 * i] * rhs.real() - lhs._data[2 * i + 1] * rhs.imag();
            ret._data[2 * i + 1] = lhs._data[2 * i] * rhs.imag() + lhs._data[2 * i + 1] * rhs.real();
        }
        return ret;
    }
    friend vec operator*(const T &lhs, const vec &rhs) {
        vec ret{};
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret._data[2 * i] = lhs.real() * rhs._data[2 * i] - lhs.imag() * rhs._data[2 * i + 1];
            ret._data[2 * i + 1] = lhs.real() * rhs._data[2 * i + 1] + lhs.imag() * rhs._data[2 * i];
        }
        return ret;
    }
    friend vec operator*(const vec &lhs, const typename unwrap<T>::type &rhs) {
        vec ret{};
        ret._data = lhs._data * rhs;
        return ret;
    }
    friend vec operator*(const typename unwrap<T>::type &lhs, const vec &rhs) {
        vec ret{};
        ret._data = lhs * rhs._data;
        return ret;
    }
    friend bool operator==(const vec &lhs, const vec &rhs) {
        return lhs == rhs;
    }
    friend bool operator==(const vec &lhs, const T &rhs) {
        bool ret = true;
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret = ret && lhs._data[2 * i] == rhs.real();
            ret = ret && lhs._data[2 * i + 1] == rhs.imag();
        }
        return ret;
    }
    friend bool operator==(const T &lhs, const vec &rhs) {
        bool ret = true;
        #pragma unroll N
        for (size_t i = 0; i < N; i++) {
            ret = ret && lhs.real() == rhs._data[2 * i];
            ret = ret && lhs.imag() == rhs._data[2 * i + 1];
        }
        return ret;
    }
    friend bool operator!=(const vec &lhs, const vec &rhs) {
        return !(lhs == rhs);
    }
    friend bool operator!=(const vec &lhs, const T &rhs) {
        return !(lhs == rhs);
    }
    friend bool operator!=(const T &lhs, const vec &rhs) {
        return !(lhs == rhs);
    }
};

} // namespace detail
} // namespace t2sp

using complexf2 = t2sp::detail::vec<complexf, 2>;
using complexf4 = t2sp::detail::vec<complexf, 4>;
using complexf8 = t2sp::detail::vec<complexf, 8>;
using complexd2 = t2sp::detail::vec<complexd, 2>;
using complexd4 = t2sp::detail::vec<complexd, 4>;
using complexd8 = t2sp::detail::vec<complexd, 8>;
