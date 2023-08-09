# `COPY`

This reference design shows how to implement the standard COPY in BLAS as defined in the [oneMKL interface](https://oneapi-src.github.io/oneMKL/domains/blas/copy.html):

$y \longleftarrow x$

where $x$ and $y$ are vectors.

The kernel is implemented by reconfiguring the [systolic array of vector addition](../reconfigurable_vecadd/README.md), where the design details and performance metrics are described.

## Build and run on Linux

Follow the general instructions [here](../README.md#user-content-build-a-kernel-and-run-on-Linux). The variations of the kernel are `scopy`, `dcopy`, `ccopy` and `zcopy`.