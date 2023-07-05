# `SRYK`

This reference design implements the standard SYRK in BLAS as defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/syrk.html):
```
C := alpha*op(A)*op(A)<sup>T</sup> + beta*C
```
or
```
C := alpha*B*A* + beta*C
```

where matrix `A` is symmetric.

The kernel is implemented by configuring the [reconfigurable matrix multiplication](../recnfigurable_matmul/README.md), where the design details and performance metrics are described.
This kernel has the following restrictions in implementation:
* Matrix storage: row-major.
* Data size: `n` and `k` must be multiples of the vectorized dimensions of the systolic array.


## Build and run on Linux

See the instruction for [blas](../README#Build-a-kernel-and-run-on-Linux). The variations of the kernel are `ssyrk`, `dsyrk`, `csyrk`, and `zsyrk`.