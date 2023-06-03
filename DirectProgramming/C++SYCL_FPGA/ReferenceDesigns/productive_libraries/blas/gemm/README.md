# `GEMM`

This reference design shows how to implement oneMKL-compatible GEMM (*General Matrix Multiply*) using the [T2SP](https://github.com/IntelLabs/t2sp) DSL and compile the DSL to oneAPI code:

* `gemm.cpp` - The implementation of GEMM using T2SP DSL.

* `test.cpp` - Some correctness tests adapted from [oneMKL's test suite](https://github.com/oneapi-src/oneMKL/blob/develop/tests/unit_tests/blas/level3/gemm.cpp), using oneMKL's GEMM as a reference.

* `hardware_demo.cpp` - A demo showing how to compile GEMM to FPGA hardware.

## Purpose

This FPGA reference design demonstrates GEMM, as defined in [oneMKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2023-1/gemm-001.html):

```
C := alpha*op(A)*op(B) + beta*C
```
where `op(X)` is one of `op(X) = X`, or `op(X) = X<sup>T</sup>`, or `op(X) = X<sup>H</sup>`, `alpha` and `beta` are scalars, and `A`, `B` and `C` are matrices.

The kernel is implemented by configuring the [reconfigurable matrix multiplication](../recnfigurable_matmul/README.md), where the design details and performance metrics are described.

## Build and run on Linux

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

For example, to build a single-precision sysotolic array for an A10 FPGA, a typical process is as follows:
   ```shell
   #


   # Generate correctness tests. This will generate OneAPI source file from the T2SP specification with tiny size on an FPGA emulator.
   make tests

   # Test for correctness
   ../bin/test_0
   ../bin/test_1
   ../bin/test_2
   ../bin/test_3
   ../bin/test_4
   ```

Now that the correctness is verified, we can go with large size for performance:
   ```shell
   # Generate OneAPI source file from the T2SP specification
   make oneapi_sgemm_large_a10

   # Generate the HTML performance report.
   make report_sgemm_large_a10

   # Synthesize a bitstream for FPGA hardware (This takes ~5 hrs).
   make synthesize_sgemm_large_a10
   ```
   These commands invoke the corresponding commands in `reconfigurable_matmul` to do the actual job. The generated OneAPI source files, report, and bitstream are located under `reconfigurable_matmul/oneapi, reports, bin`, respectively.

   ```shell
   # Generate a demo application, which is linked with the above generated bitstream.
   make demo_sgemm_large_a10

   # Demo on the hardware
   ../bin/demo_sgemm_large_a10
   ```