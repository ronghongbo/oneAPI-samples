# Matrix Multiplication

This reference design shows how to implement a matrix multiplication circuit that can be dynamically reconfigured to implement the following BLAS kernels:

* `GEMM` - Computes a matrix-matrix product with general matrices.
* `SYMM` - Computes a matrix-matrix product where one input matrix is symmetric and one matrix is general.
* `HEMM` - Computes a matrix-matrix product where one input matrix is Hermitian and one matrix is general.
* `SYRK` - Performs a symmetric rank-k update.
* `HERK` - Performs a Hermitian rank-k update.

The design is compatible with [oneMKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-0/blas-level-3-routines.html) and written in the [T2SP](https://github.com/IntelLabs/t2sp) DSL, which generates oneAPI code.

| Area                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| What you will learn | How to implement high performance matrix multiplication on an FPGA |
| Time to complete    | ~1 hr (excluding compile time)                               |
| Category            | Reference Designs and End to End                             |

## Purpose

This design demonstrates the following matrix-matrix product:

```
C := alpha*op(A)*op(B) + beta*C
```
where `op(X)` is one of `op(X) = X`, or `op(X) = X<sup>T</sup>`, or `op(X) = X<sup>H</sup>`, `alpha` and `beta` are scalars, and `A`, `B` and `C` are matrices.

By providing appropriate parameters, the above compute simulates various BLAS kernels: `sgemm`, `dgemm`, `cgemm`, `zgemm`, `ssymm`, `dsymm`, `csymm`, `zsymm`, `chemm`, `zhemm`, `ssyrk`, `dsyrk`, `csyrk`, `zsyrk`, `cherk` and `zherk`.

Static parameters: data types
Dynamic parameters: kernel types (`GEMM`, `SYMM`, ...)

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 4 sample that demonstrates a reference design.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
```
| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 (Other Linux distributions or Windows might also work, although not tested)
| Hardware             | Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA) <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software             | Intel® oneAPI DPC++/C++ Compiler, T2SP compiler

## Key Implementation Details
The algorithm employed by the reference design is a 2-dimensional systolic array  with a sophisticated I/O network.

![](figures/matmul.png)

Files:
* `gemm.cpp` - The specification of matrix multiplication in T2SP.
* `parameters.h` - Constant parameters of the systolic array, where
** `KKK` - SIMD lanes in a PE.
** `JJJ` - Columns of the systolic array.
** `III` - Rows of the systolic array.
** `JJ ` - Columns of matrix `B` to process in a PE
** `II ` - Rows of matrix `A` to process in a PE. There are `II*JJ` number of results to reduce in the PE.
** `KK ` - `KKK * KK` is the columns of matrix A / rows of matrix B to reduce in a PE.

The parameters are defined for two configurations: tiny and large. The tiny configuration specifies a 4x4 systolic array, with each PE computing 16 results. The large configuration tries to maximize the utilization of resources, and varies with precision and hardware.

* `CMakeLists.txt` - cmake targets
* `CMakeInterface.txt` - a file shared by each kernel. Building a target of a kernel may, through this interface, invokes building a target of this reconfigurable matrix multiplication.

## Metrics

The data below are with a large configuration. See `parameters.h` for details.

Single precision: on A10, 10x8 array, each PE computes 1024 results.
| Device | Logic utilization | DSP blocks | RAM blocks | Frequency | Throughput | Matrix Size |
| ------ | --------- | ---------- | ----------------- | ---------- | ---------- | -----------|
| Intel Arria 10 GX 1150   |  218,275 / 427,200 ( 51 % ) |  1,314 / 1,518 ( 87 % ) | 2,198 / 2,713 ( 81 % ) | 223 MHZ | 479 GFLOPS |
| Intel Stratix 10 SX 2800 | | | | | | |

Double precision:

Complex single precision:

Complex double precision:

## Build

1. Configure the build system for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.

   ```shell
   mkdir -p build
   cd build
   cmake ..
   ```

   For **Intel Stratix® 10 SX**, enter the following:

   ```shell
   mkdir -p build
   cd build
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```

2. Compile the design.

   ```shell
   make (oneapi|report|synthesize)_Kernel_Size_Hardware
   ```
`Kernel` is precision (`s` for single precision, `d` for double-precision, `c` for complex single-precision, or `z` for complex double-precision) with a kernel name (here `matmul`).
`Size` is `tiny` or `large`. `Hardware` is either `a10` or `s10`.

For example,

   ```shell
   # Generate OneAPI source file from the T2SP specification for single-precision matrix multiplication with a tiny systolic arrary on an A10 FPGA.
   make oneapi_smatmul_tiny_a10

   # Generate an HTML report of resource usage, frequencies, etc. for single-precision matrix multiplication with a tiny systolic arrary on an A10 FPGA.
   make report_smatmul_tiny_a10

   # Generate a bitstream for complex double-precision matrix multiplication with a large systolic arrary on an S10 FPGA.
   make synthesize_zmatmul_large_s10
   ```

The generated files are located in
** `oneapi`: the OneAPI source files generated
** `bin`: the bitstreams generated
** `reports`: the HTML files generated

## Test
Go to the directories for the kernels that this compute simulates, and follow the instructions there to test.

## Clean

To clean up files generated during making a target, use the generated `cmake_clean.cmake` file. For example,
   ```shell
   cmake -P  CMakeFiles/synthesize_zmatmul_large_s10.dir/cmake_clean.cmake
   ```
This command cleans up all the files generated during making the target `synthesize_zmatmul_large_s10`.
