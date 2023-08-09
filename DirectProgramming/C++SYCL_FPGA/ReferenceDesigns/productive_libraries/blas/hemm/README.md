# `HEMM`

This reference design implements the standard HEMM in BLAS as defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/hemm.html):

$C \longleftarrow \alpha * A * B + \beta * C$

or

$C \longleftarrow \alpha * B * A + \beta * C$

where $alpha$ and $beta$ are scalars, and $A$ is a Hermitian matrix.

The kernel is implemented by reconfiguring the [systolic array of matrix multiply](../reconfigurable_matmul/README.md), where the design details and performance metrics are described.

## Build and run on Linux

Follow the general instructions [here](../README.md#user-content-build-a-kernel-and-run-on-Linux). The variations of the kernel are `chemm` and `zhemm`.