# `SRYK`

This reference design implements the standard SYRK in BLAS as defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/syrk.html):

$C \longleftarrow \alpha * op(A) * op(A)^T+\beta * C$

where $op(X)$ is $X$ or $X^T$, $alpha$ and $beta$ are scalars, and $C$ is a symmetric matrix and $A$ is a general matrix. 

## Build and run on Linux

Follow the general instructions [here](../README.md#Build-a-kernel-and-run-on-Linux). The variations of the kernel are `ssyrk`, `dsyrk`, `csyrk`, and `zsyrk`.

Under the hood, the kernel invokes the [reconfigurable matrix multiplication](../reconfigurable_matmul/README.md), where the design details and performance metrics are described.
