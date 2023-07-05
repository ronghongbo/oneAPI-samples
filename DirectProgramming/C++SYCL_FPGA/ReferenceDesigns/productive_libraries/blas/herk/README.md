# `HERK`

This reference design implements the standard HERK in BLAS as defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/herk.html):
```
C := alpha*op(A)*op(A)<sup>H</sup> + beta*C
```

where `op(X) = X` or `X<sup>H</sup>`, and matrix `C` is Hermitian.

The kernel is implemented by configuring the systolic array of [matrix multiply](../recnfigurable_matmul/README.md), where the design details and performance metrics are described.
This kernel has the following restrictions in implementation:
* Matrix storage: row-major.
* Data size: `n` and `k` must be multiples of the vectorized dimensions of the systolic array.

## Build and run on Linux

See the instruction for [blas](../README#Build-a-kernel-and-run-on-Linux). The variations of the kernel are `cherk` and `zherk`.