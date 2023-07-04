# `HERK`

This reference design implements the standard HERK in BLAS as defined in the [oneMKL interface](https://oneapi-src.github.io/oneMKL/domains/blas/blas.html):
```
C := alpha*op(A)*op(A)<sup>H</sup> + beta*C
```

where `op(X) = X` or `X<sup>H</sup>`, and matrix `C` is Hermitian.

The kernel is implemented by configuring the systolic array of [matrix multiply](../recnfigurable_matmul/README.md), where the design details and performance metrics are described.
This kernel has the following restrictions in implementation:
* Matrix storage: row-major.
* Data size: `n` and `k` must be multiples of the vectorized dimensions of the systolic array.


## `Files`

api.hpp  bin  build  CMakeLists.txt  demo.cpp  README.md  test.cpp


* `api.hpp` - The API to invoke the kernel in any C++ application.

* `test.cpp` - Unit tests of correctness adapted from [oneMKL's test suite](https://github.com/oneapi-src/oneMKL/blob/develop/tests/unit_tests/blas/level3/herk_usm.cpp), with slight changes to respect the above restrictions.

* `demo.cpp` - A demo showing how to compile HERK to FPGA hardware.

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

2. Test correctness (with a tiny-scale systolic array on an FPGA emulator).

   ```shell
   make tests
   cd ../bin; ./test_0 && ./test_1 && ./test_2 && ./test_3; cd -
   ```

3. Test performance

    ```shell
    make (report|demo)_(cherk|zherk)_(tiny|large)_(a10|s10)
    ```
Take `cherk` for example:
   ```shell
   # Generate the HTML performance report.
   make report_cherk_large_a10

   ```shell
   # Generate a demo application. If the large-scale reconfigurable systolic array has not been built yet, it will be built, and then linked with the demo application.
   make demo_cherk_large_a10

   # Demo on the hardware
   ../bin/demo_cherk_large_a10
   ``` 