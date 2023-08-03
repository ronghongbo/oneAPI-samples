# Matrix Multiplication

This design demonstrates the following matrix-matrix product:

$C \longleftarrow \alpha * op(A) * op(B) + \beta * C$

where $op(X)$ is $X$, $X^T$, or $X^H$, $alpha$ and $beta$ are scalars, and $A$, $B$ and $C$ are matrices. 

The design yields a systolic array for each valid combination of the data types. The array is reconfigurable: by providing appropriate parameters, the array simulates the following BLAS kernels:
* `GEMM` - Computes a matrix-matrix product with general matrices.
* `SYMM` - Computes a matrix-matrix product where one input matrix is symmetric and one matrix is general.
* `HEMM` - Computes a matrix-matrix product where one input matrix is Hermitian and one matrix is general.
* `SYRK` - Performs a rank-k update of the upper or lower triangle of a symmetric matrix.
* `HERK` - Performs a rank-k update of the upper or lower triangle of a Hermitian matrix.

Note:
* `SYRK` and `HERK` are to be available in the next release

The parameters include the following:
* `TransposeA`, `ConjugateA`, `SymmetricA`, `HermitianA`, `UpA`
* `TransposeB`, `ConjugateB`, `SymmetricB`, `HermitianB`, `UpB`
* `SymmetricC`, `HermitianC`, `UpC`
* `HalfSpaceOut`
* `alpha`, `beta`

where 

* `TransposeX`, `ConjugateX`: Is matrix X to be transposed? Is it to be conjugated?
* `SymmetricX`, `HermitianX`: Is matrix X symmetric? Is it Hermitian?
* `UpX`: Given matrix X as symmetric or Hermitian, is its upper triangle stored?
* `HalfSpaceOut`: Compute only half of the output matrix? This is true when the output is symmetric or Hermitian. In this case, the systolic array computes only the upper triangle of the output, in terms of tiles. For the tiles crossing the diagonal, we ensure the correctness of only their data above or on the diagonal.
  
| Area                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| What you will learn | How to implement a high performance systolic array for matrix multiplication on an FPGA |
| Time to complete    | ~1 hr (excluding compile time)                               |
| Category            | Reference Designs and End to End                             |

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
| Software             | Intel® oneAPI DPC++/C++ Compiler 2023.2<br> BSP used for Arria® 10 FPGA: inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp<br>T2SP compiler (a beta version is pre-installed)

## The design
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
* `api.hpp` - A BLAS-style programming interface to invoke the design.

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

## Build, test, and clean
Follow the general instructions [here](../README.md#Build-a-kernel-and-run-on-Linux). Use any of the following variations of the kernels covered by the design, including `sgemm`, `dgemm`, `cgemm`, `zgemm`, `ssymm`, `dsymm`, `csymm`, `zsymm`, `chemm`, `zhemm`, `ssyrk`, `dsyrk`, `csyrk`, `zsyrk`, `cherk` and `zherk`*, and the design will be synthesized automatically. Alternatively, one can install the pre-synthesized bitstreams following the instructions there.
