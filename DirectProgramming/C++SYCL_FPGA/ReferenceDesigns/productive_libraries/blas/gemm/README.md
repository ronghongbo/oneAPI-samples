# `GEMM`

This reference design implements the standard HERK in BLAS as defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/gemm.html):

```
C := alpha*op(A)*op(B) + beta*C
```
where `op(X)` is one of `op(X) = X`, or `op(X) = X<sup>T</sup>`, or `op(X) = X<sup>H</sup>`, `alpha` and `beta` are scalars, and `A`, `B` and `C` are matrices.

The kernel is implemented by configuring the systolic array of [matrix multiply](../recnfigurable_matmul/README.md), where the design details and performance metrics are described.
This kernel has the following restrictions in implementation:
* Matrix storage: row-major.
* Data types: `s` (single-precision), `d`(double-precision), `c`(complex single-precision), `z`(complex double-precision).
* Data size: `n` and `k` must be multiples of the vectorized dimensions of the systolic array.

## Build and run on Linux

See the instruction for [blas](../README.md#Build-a-kernel-and-run-on-Linux). The variations of the kernel are `sgemm`, `dgemm`, `cgemm` and `zgemm`.
