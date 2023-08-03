# `HERK`

This reference design implements the standard HERK in BLAS as defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/herk.html):

$C \longleftarrow \alpha * op(A) * op(A)^H+\beta * C$

where $op(X)$ is $X$ or $X^H$, $alpha$ and $beta$ are real scalars, and $C$ is a Hermitian matrix and $A$ is a general matrix. 

## Build and run on Linux

Follow the general instructions [here](../README.md#Build-a-kernel-and-run-on-Linux). The variations of the kernel are  `cherk` and `zherk`.

Under the hood, the kernel invokes the [reconfigurable matrix multiplication](../reconfigurable_matmul/README.md), where the design details and performance metrics are described.
