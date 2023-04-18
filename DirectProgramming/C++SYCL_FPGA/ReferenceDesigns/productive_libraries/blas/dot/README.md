# `DOT` Sample

This reference design shows how to implement oneMKL-compatible DOT (*Dot Product*) using the [T2SP](https://github.com/IntelLabs/t2sp) DSL and compile the DSL to oneAPI code:

* `dot.cpp` - The implementation of DOT using T2SP DSL.

* `test.cpp` - Some correctness tests from [oneMKL's test suite](https://github.com/oneapi-src/oneMKL/blob/develop/tests/unit_tests/blas/level3/dot.cpp), using oneMKL's DOT as a reference.

* `hardware_demo.cpp` - A demo showing how to compile DOT to FPGA hardware.

| Area                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| What you will learn | How to implement high performance DOT on an FPGA using T2SP |
| Time to complete    | ~1 hr (excluding compile time)                               |
| Category            | Reference Designs and End to End                             |
	
## Purpose

TODO

## Prerequisites

TODO

### Performance

TODO

## Key Implementation Details

TODO

## Build the `DOT` Design

### On Linux*

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
   # Generate test, compile for emulation (fast compile time, targets emulated FPGA device).
   make dot_test
   # Generate the HTML performance report.
   make dot_report
   # Generate fpga demo, compile for FPGA hardware (longer compile time, targets FPGA device).
   make dot_fpga
   ```

## Run the `DOT` Design

### On Linux

#### Run on FPGA Emulator

```shell
./dot_test
```

#### Run on FPGA

```shell
./dot_fpga
```
