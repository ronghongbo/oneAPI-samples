# `GEMM`

This reference design implements the standard GEMM in BLAS as defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/gemm.html):

$C \longleftarrow \alpha * op(A) * op(B) + \beta * C$

where $op(X)$ is $X$, $X^T$, or $X^H$, $\alpha$ and $\beta$ are scalars, and $A$, $B$ and $C$ are matrices.

The kernel is implemented by reconfiguring the [systolic array of matrix multiply](../reconfigurable_matmul/README.md), where the design details and performance metrics are described.

## Build and run on Linux

Follow the general instructions for [blas](../README.md#user-content-build-a-kernel-and-run-on-Linux). The variations of this kernel are `sgemm`, `dgemm`, `cgemm` and `zgemm`.
