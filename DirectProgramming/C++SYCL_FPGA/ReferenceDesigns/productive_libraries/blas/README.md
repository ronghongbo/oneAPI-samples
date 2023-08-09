# `BLAS`

This directory contains FPGA reference designs for the standard BLAS kernels defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/blas.html). The row-major, USM-based SYCL interface is supported.

To reduce engineering efforts, kernels with similar computes are grouped and generalized into a single systolic array so that the array can be dynamically reconfigured to simulate all the kernels, without losing performance. Below are the kernels currently supported, with one table for one group:

## `Level 1 kernels`

| Kernel            | Formula                                           | Description                                                                                                                                |
| ----------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| $\mathbf{dot}$    | $\vec{X}\cdot \vec{Y}$                            | Dot product.|
| $\mathbf{sdsdot}$ | $sb+\vec{X}\cdot \vec{Y}$                         | A dot product between two single-precision vectors , plus a single-precision float $sb$                                                    |
| $\mathbf{dotc}$   | $\overline{\vec{X}}\cdot \vec{Y}$                 | A dot product between two complex vectors, conjugating the first of them                                                                   |
| $\mathbf{dotu}$   | $\vec{X}\cdot \vec{Y}$                            | A dot product between two complex vectors                                                                                                  |
| $\mathbf{nrm2}$   | $\|\vec{X}\|$                                     | Euclidean norm of a vector                                                                                                                 |
| $\mathbf{asum}$   | $\sum_{i=0}^{N}(\mid Re(x_i)\mid+\mid Im(x_i)\mid)$ | Sum of the magnitudes of elements                                                                                                          |

| Kernel            | Formula                                           | Description                                                                                                                                |
| ----------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| $\mathbf{axpy}$   | $\alpha\vec{X}+\vec{Y}$                           | Vector addition                                                                                                                            |
| $\mathbf{scal}$   | $\alpha\vec{X}$                                   | Scalar Multiplication of Vector                                                                                                            |
| $\mathbf{copy}$   | $\vec{Y}\leftarrow\vec{X}$                        | Copy a vector                                                                                                                              |

## `Level 3 kernels`

 Kernel          | Formula             | Description       |
| --------------- | ------------------- | ----------|
| $\mathbf{gemm}$ | $\alpha * op(A) * op(B)+\beta * C$ |Multiplication of general matrices. $op(X)$ is one of $X$, $X^T$, and $X^H$ |
| $\mathbf{symm}$ | $\alpha * A* B+\beta * C$, or  $\alpha * B * A+\beta * C$ | A is a symmetric matrix |
| $\mathbf{hemm}$ |$\alpha * A * B+\beta * C$, or  $\alpha * B * A+\beta * C$ | A is a Hermitian matrix |
| $\mathbf{syrk}$ | $C \leftarrow \alpha * op(A) * op(A)^T + \beta * C$ |$op(X)=X$ or $op(X) = X^T$, C is a symmtric matrix. |
| $\mathbf{herk}$ | $C \leftarrow \alpha * op(A) * op(A)^H + \beta * C$ |$op(X)=X$ or $op(X) = X^H$, C is a Hermitian matrix. |

Note:
* syrk and herk are to be available in the next release.

## `File structure`

All the kernels are put under the `blas` directory. Every kernel has the following files under it:

* `api.hpp` - The API to invoke the kernel in any SYCL application.
* `test.cpp` - Unit tests for correctness, adapted from [oneMKL's test suite](https://github.com/oneapi-src/oneMKL/blob/develop/tests/unit_tests/blas/), with slight changes to respect the row-major layout and the sizes of the reconfigurable systolic arrays.
* `demo.cpp` - Demonstrating how to invoke the kernel on a real FPGA hardware.
* `CMakeLists.txt` - A cmake script.
* `README.md` - A short description of the kernel.

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

    Each kernel usually has several variations, depending on the the precision. For example, `gemm` has 4 precisions supported,  `s` (single-precision), `d`(double-precision), `c`(complex single-precision), and `z`(complex double-precision), and correspondingly, 4 variations: `sgemm`, `dgemm`, `cgemm` and `zgemm`.

   ```shell
   # Replace the VARIATION below with a specific variation of the kernel.
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

T2SP (Temporal To Spatial Programming, previously called T2S) constructs systolic arrays for dense tensor computes. A stable open source is available [here](https://github.com/IntelLabs/t2sp). For convenience, a latest experimental version of the T2SP compiler, namely Lasa, has been pre-installed in this website, and is automatically invoked when making tests or a demo.

The key point is to express a systolic array as two dataflow graphs, one for the compute, one for data movement. To understand more details, please read this latest publication:

1. Lasa: Abstraction and Specialization for Productive and Performant Linear Algebra on FPGAs. Xiaochen Hao, Mingzhe Zhang, Ce Sun, Zhuofu Tao, Hongbo Rong, Yu Zhang, Lei He, Eric Petit, Wenguang Chen, Yun Liang. FCCM, 2023. https://ieeexplore.ieee.org/document/10171577.

