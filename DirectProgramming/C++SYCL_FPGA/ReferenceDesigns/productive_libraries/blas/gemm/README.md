# `GEMM` Sample

This sample is a reference design that shows how to implement MKL-compatible GEMM (*General Matrix Multiply*) using the [T2SP](https://github.com/IntelLabs/t2sp) DSL and compile the DSL to oneAPI code using the [T2SP](https://github.com/IntelLabs/t2sp) compiler:

* `gemm.cpp` - The implementation of GEMM using T2SP DSL.

* `test.cpp` - Some correctness tests from [oneMKL's test suite](https://github.com/oneapi-src/oneMKL/blob/develop/tests/unit_tests/blas/level3/gemm.cpp), using oneMKL's GEMM as reference.

* `hardware_demo.cpp` - A demo showing how to use GEMM and compile it to FPGA hardware.

| Area                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| What you will learn | How to implement high performance GEMM on FPGA using T2SP |
| Time to complete    | ~1 hr (excluding compile time)                            |

## Purpose

TODO

## Prerequisites

TODO

### Performance

TODO

### Correctness

TODO

## Key Implementation Details

TODO

## Build the `GEMM` Design

### On Linux*

1. Change to T2SP directory.

2. Set some environment variables using the script provided by T2SP.
   
   ```shell
   # working on intel devcloud
   source setenv.sh devcloud oneapi fpga
   # working locally
   source setenv.sh local oneapi fpga
   ```

3. Change to the sample directory.

4. Configure the build system for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.
   
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

5. Compile the design. Here we first compile `gemm.cpp` and then use it to generate oneapi code and then compile the generated code to FPGA. These steps have been written in `CMakeLists.txt`, we only need to enter the following:
   
   ```shell
   # Generate test, compile for emulation (fast compile time, targets emulated FPGA device).
   make gemm_test
   # Generate the HTML performance report.
   make gemm_report
   # Generate fpga demo, compile for FPGA hardware (longer compile time, targets FPGA device).
   make gemm_fpga
   ```

### On Windows

TODO

## Run the `QRD` Design

### On Linux

#### Run on FPGA Emulator

```shell
./gemm_test_0 && ./gemm_test_1 && ./gemm_test_2 && ./gemm_test_3 && ./gemm_test_4
```

#### Run on FPGA

```shell
./gemm_fpga
```

### On Windows

TODO
