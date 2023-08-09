# `SCAL`

This reference design shows how to implement the standard SCAL in BLAS as defined in the [oneMKL interface](https://oneapi-src.github.io/oneMKL/domains/blas/scal.html):

$x \longleftarrow \alpha * x$

where $x$ is a vector and $\alpha$ is scalar.

Restriction:

* Currently, the data type of $x$'s elements and the data type of $\alpha$ need be the same. This restriction is to be removed in the next release.

The kernel is implemented by reconfiguring the [systolic array of vector addition](../reconfigurable_vecadd/README.md), where the design details and performance metrics are described.

## Build and run on Linux

Follow the general instructions [here](../README.md#user-content-build-a-kernel-and-run-on-Linux). The variations of the kernel are `sscal`, `dscal`, `cscal` and `zscal`.
