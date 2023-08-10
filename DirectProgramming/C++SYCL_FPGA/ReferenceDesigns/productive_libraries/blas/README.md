# `BLAS`

This directory contains FPGA reference designs for the standard BLAS kernels defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/blas.html). The row-major, USM-based SYCL interface is supported.

Kernels of similar computes are grouped and generalized into a single systolic array so that the array can be dynamically reconfigured to simulate all the kernels, minimizing maintenance cost without losing performance. Below are the kernels supported in this release:

## `Level 1 kernels`

A [dot-product systolic array](reconfigurable_dotprod/README.md) supports

| Kernel            | Formula                                           | Description                                                                                                                                | VARIATION |
| ----------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----|
| [dot](https://oneapi-src.github.io/oneMKL/domains/blas/dot.html)    | $\vec{X}\cdot \vec{Y}$                            | Dot product.| sdot, ddot, dsdot |  | 
| [sdsdot](https://oneapi-src.github.io/oneMKL/domains/blas/sdsdot.html) | $sb+\vec{X}\cdot \vec{Y}$                         | Return a single-precision result with a dot product of two vectors accumulated in double-precision | sdsdot|
| [dotc](https://oneapi-src.github.io/oneMKL/domains/blas/dotc.html)   | $\overline{\vec{X}}\cdot \vec{Y}$                 | A dot product between two complex vectors, conjugating the first of them     | cdotc, zdotc|
| [dotu](https://oneapi-src.github.io/oneMKL/domains/blas/dotu.html)   | $\vec{X}\cdot \vec{Y}$                            | A dot product between two complex vectors                               | cdotu, zdotu|
| [nrm2](https://oneapi-src.github.io/oneMKL/domains/blas/nrm2.html)   | $\parallel \vec{X} \parallel$                                     | Euclidean norm of a vector                              | snrm2, dnrm2, scnrm2, dznrm2 |
| [asum](https://oneapi-src.github.io/oneMKL/domains/blas/asum.html)   | sum of $\mid Re(x_i)\mid+\mid Im(x_i)\mid, \forall i$ | Sum of the magnitudes of elements                                   | sasum, dasum, scasum, dzasum |

The `VARIATION` column shows the variations of each kernel, usually the kernel name prefixed by the output/input data types. A data type can be `s` (single-precision), `d`(double-precision), `c`(complex single-precision) or `z`(complex double-precision).

A [vector-addition systolic array](reconfigurable_vecadd/README.md) supports
| Kernel            | Formula                                           | Description                                                                                                                                |VARIATION |  Note |
| ----------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |-----| --- |
| [axpy](https://oneapi-src.github.io/oneMKL/domains/blas/axpy.html)   | $\alpha * \vec{X}+\vec{Y}$                           | Vector addition                                                 | saxpy, daxpy, caxpy, zaxpy ||
| [scal](https://oneapi-src.github.io/oneMKL/domains/blas/scal.html)   | $\alpha * \vec{X}$                                   | Scale a vector                                                  | sscal, dscal, cscal, zscal | csscal and zdscal are to be supported in the next release |
| [copy](https://oneapi-src.github.io/oneMKL/domains/blas/copy.html)   | $\vec{Y}\leftarrow\vec{X}$                        | Copy a vector                                                      | scopy, dcopy, ccopy, zcopy | |

## `Level 3 kernels`

A [matrix-multiply systolic array](reconfigurable_matmul/README.md) supports
 Kernel          | Formula             | Description       |VARIATION | Note |
| --------------- | ------------------- | ----------|-----|---|
| [gemm](https://oneapi-src.github.io/oneMKL/domains/blas/gemm.html) | $\alpha * op(A) * op(B)+\beta * C$ |Multiplication of general matrices. $op(X)$ is one of $X$, $X^T$, and $X^H$ | sgemm, dgemm, cgemm, zgemm|Half and bfloat16 are to be supported in future|
| [symm](https://oneapi-src.github.io/oneMKL/domains/blas/symm.html) | $\alpha * A* B+\beta * C$, or  $\alpha * B * A+\beta * C$ | A is a symmetric matrix | ssymm, dsymm, csymm, zsymm ||
| [hemm](https://oneapi-src.github.io/oneMKL/domains/blas/hemm.html) |$\alpha * A * B+\beta * C$, or  $\alpha * B * A+\beta * C$ | A is a Hermitian matrix | chemm, zhemm ||
| syrk | $C \leftarrow \alpha * op(A) * op(A)^T + \beta * C$ |$op(X)=X$ or $op(X) = X^T$, C is a symmtric matrix. | ssyrk, dsyrk, csyrk, zsyrk| To be available in the next release |
| herk | $C \leftarrow \alpha * op(A) * op(A)^H + \beta * C$ |$op(X)=X$ or $op(X) = X^H$, C is a Hermitian matrix. |cherk, zherk| To be available in the next release |

### Restrictions of the systolic arrays

* Matrix storage: row-major.
* Data types: `s`, `d`, `c`, `z`.
* Data sizes: For memory efficiency, vectors/matrices must be loaded and stored in vectors from/to the device memory. Therefore, dimensions of the vectors/matrices must be multiples of the vector lengths. This restriction is to be removed in the next release.

## `File structure`

All the kernels are put under the `blas` directory. Every kernel has the following files under it:

* `api.hpp` - The API to invoke the kernel in any SYCL application.
* `test.cpp` - Unit tests for correctness, adapted from [oneMKL's test suite](https://github.com/oneapi-src/oneMKL/blob/develop/tests/unit_tests/blas/), with slight changes to respect the row-major layout and the sizes of the reconfigurable systolic arrays.
* `demo.cpp` - Demonstrating how to invoke the kernel on a real FPGA hardware.
* `CMakeLists.txt` - A cmake script.

The shared systolic arrays (named as `reconfigurable-*`) are also under the `blas` directory. Every systolic array has the following files under it:

* `api.hpp` - The API to invoke the systolic array.
* `spec.cpp`: A specification of the systolic array and related optimizations in a spatial programming language called [T2SP](#user-content-reference). From this specification, SYCL files will be generated by a pre-installed T2SP compiler. The SYCL files are then synthesized into a bitstream for an FPGA hardware.
* `parameters.h` : Sizes of the systolic array.
* `CMakeLists.txt` - A cmake script.
* `README.md` - A short description of the systolic array.

## Build a kernel and run on Linux

1. Configure the build system.

   ```shell
   cd KERNEL # Replace this with the specific kernel's name, e.g. gemm
   mkdir -p build
   cd build
   ```

    For Intel Arria® 10 GX FPGA:

   ```shell
   cmake ..
   ```

    For **Intel Stratix® 10 SX**:

   ```shell
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```

2. Test correctness.

   ```shell
   make tests
   ../bin/tests.sh
   ```

    Each test builds a tiny-scale systolic array and runs on an FPGA emulator.

3. Test performance

   ```shell
   # Replace the VARIATION below with a specific variation of the kernel as listed in the tables above.
   make demo_VARIATION_(tiny|large)_(a10|s10)
   ../bin/demo_VARIATION_(tiny|large)_(a10|s10)
   ```
    The demo application invokes a systolic array. If the array has not been synthesized, the above command will synthesize it from scratch automatically. Optionally, to avoid synthesis, a pre-generated bitstream for the array can be installed; the informational files including SYCL file and reports for the array, as well as a demo application, are installed too:
   ```
   ../../install_pre_gen.sh VARIATION_(tiny|large)_(a10|s10)
   ```

    Take `sgemm` for example:

   ```shell
   # Optional: install the pre-generated bitstream.
   ../../install_pre_gen.sh sgemm_large_a10

   # Generate a demo application on the FPGA hardware
   make demo_sgemm_large_a10


   # Demo on the hardware
   ../bin/demo_sgemm_large_a10
   ```

    Known issue: if there is an error message like "Error writing bitstream to FPGA: reconfiguration error" on A10, it might be due to the security feature of devstack 1.2.1. Try to unsign and re-run the binary like this:
    ```shell
    # Unsign the bitstream
    make unsign_VARIATION_(tiny|large)_a10

    # Demo on the hardware
    ../bin/demo_VARIATION_(tiny|large)_a10.unsigned
   ```
    For example:
    ```shell
    make unsign_sgemm_large_a10
    ../bin/demo_sgemm_large_a10.unsigned
   ```

4. Delete all generated files for a kernel

   ```shell
   # Replace the VARIATION below with a specific variation of the kernel.
   make clean_VARIATION_(tiny|large)_(a10|s10)
   ```
       For example:
    ```shell
    make clean_sgemm_large_a10
   ```

# Reference

[T2SP](https://github.com/IntelLabs/t2sp) (Temporal To Spatial Programming, previously called T2S) is a productive language to construct systolic arrays for dense tensor computes. For convenience, a latest experimental version of the T2SP compiler, namely Lasa, has been pre-installed in this website, and is automatically invoked when making tests or a demo.

The key point of the language is to express a systolic array as two dataflow graphs, one for compute, the other for data movement. To understand more details, please read

1. Lasa: Abstraction and Specialization for Productive and Performant Linear Algebra on FPGAs. Xiaochen Hao, Mingzhe Zhang, Ce Sun, Zhuofu Tao, Hongbo Rong, Yu Zhang, Lei He, Eric Petit, Wenguang Chen, Yun Liang. FCCM, 2023. https://ieeexplore.ieee.org/document/10171577.

