# `BLAS`

This directory contains FPGA reference designs for the standard BLAS kernels defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/blas.html). The row-major, USM-based SYCL interface is supported.

To reduce engineering efforts, kernels with similar computes are grouped so that a single systolic array can be built and reconfigured to cover each kernel in the group, without losing performance. Below are the kernels supported, with one table for one group:

## `Level 1 kernels`

| Kernel            | Formula                                           | Data types of (inputs, output) | Description                                                                                                                                |
| ----------------- | ------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| $\mathbf{dot}$    | $\vec{X}\cdot \vec{Y}$                            | (s, s), (d, d), (s, d)         | Dot product. For the mixed precision version (inputs are float while result is double), the dot product is computed with double precision. |
| $\mathbf{sdsdot}$ | $sb+\vec{X}\cdot \vec{Y}$                         | (s, s)                         | A dot product between two single-precision vectors , plus a single-precision float $sb$                                                    |
| $\mathbf{dotc}$   | $\overline{\vec{X}}\cdot \vec{Y}$                 | (c, c), (z, z)                 | A dot product between two complex vectors, conjugating the first of them                                                                   |
| $\mathbf{dotu}$   | $\vec{X}\cdot \vec{Y}$                            | (c, c), (z, z)                 | A dot product between two complex vectors                                                                                                  |
| $\mathbf{nrm2}$   | $\|\vec{X}\|$                                     | (s, s), (d, d), (c, s),(z, d)  | Euclidean norm of a vector                                                                                                                 |
| $\mathbf{axpy}$   | $\alpha\vec{X}+\vec{Y}$                           | (s, s),(d, d),(c, c),(z, z)    | Vector addition                                                                                                                            |
| $\mathbf{scal}$   | $\alpha\vec{X}$                                   | (s, s),(d, d),(c, c),(z, z)    | Scalar Multiplication of Vector                                                                                                            |
| $\mathbf{copy}$   | $\vec{Y}\leftarrow\vec{X}$                        | (s, s),(d, d),(c, c),(z, z)    | Copy a vector                                                                                                                              |
| $\mathbf{asum}$   | $\sum_{i=0}^N(\mid Re(x_i)\mid+\mid Im(x_i)\mid)$ | (s, s),(d, d),(c, c),(z, z)    | Sum of the magnitudes of elements                                                                                                          |

## `Level 2 kernels`

## `Level 3 kernels`

 Kernel          | Formula             | Data types | Description       |
| --------------- | ------------------- | ----------|------- |
| $\mathbf{gemm}$ | $\alpha op(A)op(B)+\beta C$ | s, d, c, z  | Multiplication of general matrices. $op(X)$ is one of $X$, $X^T$, and $X^H$ |
| $\mathbf{symm}$ | $\alpha AB+\beta C$, or  $\alpha BA+\beta C$ |s, d, c, z | A is a symmetric matrix |
| $\mathbf{hemm}$ |$\alpha AB+\beta C$, or  $\alpha BA+\beta C$ |s, d, c, z | A is a Hermitian matrix |
| $\mathbf{syrk}$ | $C \leftarrow \alpha op(A)op(A)^T + \beta C$ | s, d, c, z |$op(X)=X$ or $op(X) = X^T$, C is a symmtric matrix. |
| $\mathbf{herk}$ | $C \leftarrow \alpha op(A)op(A)^H + \beta C$ | c/z for the matrices, with s/d for the scalars |$op(X)=X$ or $op(X) = X^H$, C is a Hermitian matrix. |

## `File structure`

All the kernels are put under the `blas` directory. Every kernel has the following files under it:

* `api.hpp` - The API to invoke the kernel in any SYCL application.
* `test.cpp` - Unit tests for correctness, adapted from [oneMKL's test suite](https://github.com/oneapi-src/oneMKL/blob/develop/tests/unit_tests/blas/), with slight changes to respect the row-major layout and the sizes of the reconfigurable systolic arrays.
* `demo.cpp` - Demonstrating how to invoke the kernel.
* `CMakeLists.txt` - A cmake script.
* `README.md` - A short description of the kernel.

The shared systolic arrays (named as `reconfigurable-*`) are also under the `blas` directory. Every systolic array has the following files under it:

* `api.hpp` - The API to invoke the systolic array.
* `spec.cpp`: A brief description of the systolic array and other optimizations for it in the [T2SP language](https://github.com/IntelLabs/t2sp). From this specification, SYCL files will be generated by the T2SP compiler. The SYCL files are then synthesized into a bitstream for an FPGA hardware.
* `parameters.h` : Sizes and data types of the systolic array.
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
