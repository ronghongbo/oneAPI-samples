#pragma once
#include <complex>
#include <array>

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
    vec sqrt() const {
        vec ret{};
        for (size_t i = 0; i < N; i++) ret._arr[i] = std::sqrt(_arr[i]);
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
}
}

using complexf = std::complex<float>;
using complexd = std::complex<double>;
using complexf2 = t2sp::detail::vec<complexf, 2>;
using complexf4 = t2sp::detail::vec<complexf, 4>;
using complexf8 = t2sp::detail::vec<complexf, 8>;
using complexf16 = t2sp::detail::vec<complexf, 16>;
using complexd2 = t2sp::detail::vec<complexd, 2>;
using complexd4 = t2sp::detail::vec<complexd, 4>;
using complexd8 = t2sp::detail::vec<complexd, 8>;
using complexd16 = t2sp::detail::vec<complexd, 16>;
