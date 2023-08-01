#pragma once
#include <complex>
#include <array>

using complexf = std::complex<float>;
using complexd = std::complex<double>;

namespace t2sp {
namespace detail {
template <typename T, size_t N>
class vec {
    std::array<T, N> _arr;
  public:
    vec() = default;
    vec(const vec &) = default;
    vec(vec &&) = default;
    vec &operator=(const vec &) = default;
    vec &operator=(vec &&) = default;
    vec(const T &arg) {
        for (size_t i = 0; i < N; i++) _arr[i] = arg;
    }
    template <typename ...Args>
    vec(const Args &...args) : _arr{{args...}} {}
    vec &operator=(const T& arg) {
        for (size_t i = 0; i < N; i++) _arr[i] = arg;
        return *this;
    }
    T &operator[](size_t n) {
        return _arr[n];
    }
    const T &operator[](size_t n) const {
        return _arr[n];
    }
    constexpr static size_t size() noexcept {
        return N;
    }
    vec &operator+=(const vec &rhs) {
        for (size_t i = 0; i < N; i++) _arr[i] += rhs._arr[i];
        return *this;
    }
    vec &operator+=(const T &rhs) {
        for (size_t i = 0; i < N; i++) _arr[i] += rhs;
        return *this;
    }
    vec &operator-=(const vec &rhs) {
        for (size_t i = 0; i < N; i++) _arr[i] -= rhs._arr[i];
        return *this;
    }
    vec &operator-=(const T &rhs) {
        for (size_t i = 0; i < N; i++) _arr[i] -= rhs;
        return *this;
    }
    vec &operator*=(const vec &rhs) {
        for (size_t i = 0; i < N; i++) _arr[i] *= rhs._arr[i];
        return *this;
    }
    vec &operator*=(const T &arg) {
        for (size_t i = 0; i < N; i++) _arr[i] *= arg;
        return *this;
    }
    vec conj() const {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = std::conj(_arr[i]);
        return ret;
    }
    friend vec operator+(const vec &arg) {
        return arg;
    }
    friend vec operator-(const vec &arg) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = -arg._arr[i];
        return ret;
    }
    friend vec operator+(const vec &lhs, const vec &rhs) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = lhs._arr[i] + rhs._arr[i];
        return ret;
    }
    friend vec operator+(const vec &lhs, const T &rhs) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = lhs._arr[i] + rhs;
        return ret;
    }
    friend vec operator+(const T &lhs, const vec &rhs) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = lhs + rhs._arr[i];
        return ret;
    }
    friend vec operator-(const vec &lhs, const vec &rhs) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = lhs._arr[i] - rhs._arr[i];
        return ret;
    }
    friend vec operator-(const vec &lhs, const T &rhs) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = lhs._arr[i] - rhs;
        return ret;
    }
    friend vec operator-(const T &lhs, const vec &rhs) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = lhs - rhs._arr[i];
        return ret;
    }
    friend vec operator*(const vec &lhs, const vec &rhs) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = lhs._arr[i] * rhs._arr[i];
        return ret;
    }
    friend vec operator*(const vec &lhs, const T &rhs) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = lhs._arr[i] * rhs;
        return ret;
    }
    friend vec operator*(const T &lhs, const vec &rhs) {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = lhs * rhs._arr[i];
        return ret;
    }
    friend bool operator==(const vec &lhs, const vec &rhs) {
        bool ret = true;
        for (size_t i = 0; i < N; i++) ret = ret && lhs._arr[i] == rhs._arr[i];
        return ret;
    }
    friend bool operator==(const vec &lhs, const T &rhs) {
        bool ret = true;
        for (size_t i = 0; i < N; i++) ret = ret && lhs._arr[i] == rhs;
        return ret;
    }
    friend bool operator==(const T &lhs, const vec &rhs) {
        bool ret = true;
        for (size_t i = 0; i < N; i++) ret = ret && lhs == rhs._arr[i];
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

template<size_t N>
vec<complexf, N> operator*(const vec<complexf, N> &lhs, const float &rhs) {
    vec<complexf, N> ret{};
    for (size_t i = 0; i < N; i++) ret[i] = lhs[i] * rhs;
    return ret;
}

template<size_t N>
vec<complexf, N> operator*(const float &lhs, const vec<complexf, N> &rhs) {
    return rhs * lhs;
}

template<size_t N>
vec<complexd, N> operator*(const vec<complexd, N> &lhs, const double &rhs) {
    vec<complexd, N> ret{};
    for (size_t i = 0; i < N; i++) ret[i] = lhs[i] * rhs;
    return ret;
}

template<size_t N>
vec<complexd, N> operator*(const double &lhs, const vec<complexd, N> &rhs) {
    return rhs * lhs;
}

} // namespace detail
} // namespace t2sp

using complexf2 = t2sp::detail::vec<complexf, 2>;
using complexf4 = t2sp::detail::vec<complexf, 4>;
using complexf8 = t2sp::detail::vec<complexf, 8>;
using complexf16 = t2sp::detail::vec<complexf, 16>;
using complexd2 = t2sp::detail::vec<complexd, 2>;
using complexd4 = t2sp::detail::vec<complexd, 4>;
using complexd8 = t2sp::detail::vec<complexd, 8>;
using complexd16 = t2sp::detail::vec<complexd, 16>;