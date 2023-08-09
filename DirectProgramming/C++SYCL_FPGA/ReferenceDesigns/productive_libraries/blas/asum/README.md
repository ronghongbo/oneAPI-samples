# `ASUM`

This reference design shows how to implement the standard ASUM in BLAS as defined in the [oneMKL interface](https://oneapi-src.github.io/oneMKL/domains/blas/asum.html):

sum $\mid Re(x_i)\mid+\mid Im(x_i)\mid, \forall i$

where $x$ is a vector.

The kernel is implemented by reconfiguring the [systolic array of dot product](../reconfigurable_dotprod/README.md), where the design details and performance metrics are described.

## Build and run on Linux

Follow the general instructions [here](../README.md#user-content-build-a-kernel-and-run-on-Linux). The variations of the kernel are `sasum`, `dasum`, `casum` and `zasum`.
