# `SYMM`

This reference design implements the standard HERK in BLAS as defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/symm.html):

```
C := alpha*A*B + beta*C
```
or
```
C := alpha*B*A* + beta*C
```

where matrix `A` is symmetric.

The kernel is implemented by configuring the systolic array of [matrix multiply](../recnfigurable_matmul/README.md), where the design details and performance metrics are described.
This kernel has the following restrictions in implementation:
* Matrix storage: row-major.
* Data size: `n` and `k` must be multiples of the vectorized dimensions of the systolic array.

## Build and run on Linux

See the instruction for [blas](../README#Build-a-kernel-and-run-on-Linux). The variations of the kernel are `ssymm`, `dsymm`, `csymm` and `zsymm`.
