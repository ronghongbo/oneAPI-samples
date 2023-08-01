# `SDSDOT`

This reference design shows how to implement the standard SDSDOT in BLAS as defined in the [oneMKL interface](https://oneapi-src.github.io/oneMKL/domains/blas/blas.html) with the following restrictions:
* Matrix storage: row-major.
* Data types: `s` (single-precision), `d`(double-precision).
* Data size: `n` must be multiples of the vectorized dimensions of the systolic array.

The design is written in the [T2SP](https://github.com/IntelLabs/t2sp) DSL, which generates oneAPI code:

* `sdsdot.cpp` - The implementation of SDSDOT using T2SP DSL.

* `test.cpp` - Some correctness tests adapted from [oneMKL's test suite](https://github.com/oneapi-src/oneMKL/blob/develop/tests/unit_tests/blas/level3/sdsdot_usm.cpp), using oneMKL's SDSDOT as a reference.

* `demo.cpp` - A demo showing how to compile SDSDOT to FPGA hardware.

## Purpose

This FPGA reference design demonstrates SDSDOT:

```
C := X*Y
```
where `X`, `Y` are vectors.

The kernel is implemented by configuring the [reconfigurable vector dot](../reconfigurable_dotprod/README.md), where the design details and performance metrics are described.

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
   ../bin/test
   ```

Now that the correctness is verified, we can go with large size for performance:
   ```shell
   # Generate OneAPI source file from the T2SP specification
   make oneapi_sdsdot_large_a10

   # Generate the HTML performance report.
   make report_sdsdot_large_a10

   # Synthesize a bitstream for FPGA hardware (This takes ~5 hrs).
   make synthesize_sdsdot_large_a10
   ```
   These commands invoke the corresponding commands in `reconfigurable_dotprod` to do the actual job. The generated OneAPI source files, report, and bitstream are located under `reconfigurable_dotprod/oneapi, reports, bin`, respectively.

   ```shell
   # Generate a demo application, which is linked with the above generated bitstream.
   make demo_sdsdot_large_a10

   # Demo on the hardware
   ../bin/demo_sdsdot_large_a10
   ```