# Dot Product

This design demonstrates the following dot product between two vectors:

$$
result \longleftarrow op_3(op_1(\vec{x})\cdot op_2(\vec{y}))
$$

where $op_1(\vec{x})$ is $\vec{x}$ or $\overline{\vec{x}}$ , $op_2(y)$ is $\vec{y}$ or $\text{sign}(\vec{y})$ , $op_3(v)$ is $v$ or $\sqrt{v}$, and $\vec{x}$ and $\vec{y}$ are vectors. The definition of $\text{sign}$ is as follows

* if $\vec{y}$ is a real vector, $\text{sign}(\vec{y})$ results in a real vector, and $\text{sign}(\vec{y})_i$ is the sign bit of $y_i$.

* if $\vec{y}$ is a complex vector, $\text{sign}(\vec{y})$ results in a complex vector, and $Re(\text{sign}(\vec{y})_i)$  is the sign bit of $Re(\vec{y}_i)$ , $Im(\text{sign}(\vec{y})_i)$ is the inverse of $Im(y_i)$ 's sign bit.

The design has static and dynamic parameters. The static parameters include

* data types: the type of the inputs and result, denoted `ITYPE` and `TTYPE` respectively, A data type can be any of `s` (single precision), `d` (double precision), `c` (complex single precision), and `z` (complex double precision).

* [sizes of the systolic array](#user-content-sizes-of-a-systolic-array) that is expressed by the design.

For each combination of the static parameters, the design needs to be synthesized once.

Once the design is synthesized, the dynamic parameters are passed in and control its execution:

* `ConjugateX`, `IncX`

* `SignBitY`, `IncY`

* `SqrtRet`

where

* `ConjugateX`: is vector X to be conjugated?

* `SignBitY`: is vector Y to be applied with the `sign` function mentioned above?

* `SqrtRet`: is the result to be square-rooted?

* `IncX`, `IncY`: strides of the input vectors.

Through APIs that provide appropriate dynamic parameters and post-processing, a synthesized design simulates the following standard BLAS kernels:

* `DOT` - Computes the dot product of two real vectors.

* `DOTU` - Computes the dot product of two complex vectors.

* `DOTC` - Computes the dot product of two complex vectors, conjugating the first vector.

* `SDSDOT` - Computes a vector-vector dot product with double precision.

* `NRM2` - Computes the Euclidean norm of a vector.

* `ASUM` - Computes the sum of magnitudes of the vector elements.

| Area                | Description                                                                   |
| ------------------- | ----------------------------------------------------------------------------- |
| What you will learn | How to implement a high performance systolic array for dot product on an FPGA |
| Time to complete    | ~1 hr (excluding compile time)                                                |
| Category            | Reference Designs and End to End                                              |

## Prerequisites

| Optimized for | Description                                                                                                                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OS            | Ubuntu* 18.04/20.04 (The design is not really specific to any OS. Other Linux distributions or Windows might also work, although not tested)                                                          |
| Hardware      | Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA)<br/>Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) |
| Software      | Intel® oneAPI DPC++/C++ Compiler 2023.2<br/>BSP used for Arria® 10 FPGA: inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp<br/>T2SP compiler (a beta version is pre-installed)                  |

## The design

In this design, the input vectors are pre-processed on the host so that the FPGA device loads data sequentially from the device DRAM. This ensures that the memory accesses won't be a bottleneck of the performance. In pre-processing, the host reads an input vector $V$ and apply $op_1/op_2$ to it. The resulting $op_1(\vec{x})$ and $op_2(\vec{y})$ are sent to the device DRAM sequentially, and their inner product is computed by a linear systolic array on the device.

The systolic array fetches the vectors $op_1(\vec{x})$ and $op_2(\vec{y})$ in parts. Each PE (processing element) of the array computes the inner product of one part, and stores the result in rotating registers. Finally these registers are summed up to get the final result.

When the length of the input vector is not a multiple of the number of PEs, zeros are automatically inserted. This is zero-padding.

### Sizes of a systolic array

* `KKK` - SIMD lanes in a PE: every cycle, the PE computes a dot product, in a vectorized way, between `KKK` numbers of data from $op_1(\vec{x})$ and `KKK` numbers of data from $op_2(\vec{y})$.

* `KK` - The number of PEs.

#### Restrictions

* Data sizes: For memory efficiency, the vectors must be loaded in vectors from the device memory. Therefore, the width of $op_1(\vec{x})$ and $op_2(\vec{y})$ must be multiples of  `KKK`.

The [parameters.h](./parameters.h) file pre-defines the sizes for a tiny and large systolic array. The tiny configuration specifies a systolic array with 4 PEs. The large configuration tries to maximally utilize resources, and varies with precision and hardware. One can modify these parameters. If so, please remember to modify the `get_systolic_array_dimensions()` function in [api.hpp](./api.hpp) accordingly.

## Build and test

Follow the [general instructions](../README.md#user-content-build-a-kernel-and-run-on-Linux) to build a demo application `demo_VARIATION_SIZE_HW`for any kernel `VARIATION` that is covered by the design with a systolic array of any `SIZE` (`tiny` or `large`) on any `HW` (`a10` or `s10`), and the design will be synthesized under the hood into an image and  linked with that kernel. The correspondence between VARIATION and image, and the current status, are as follows:

| VARIATION of a kernel      | Image      | Correctness | Performance |
| -------------------------- | ---------- | ----------- | ----------- |
| sdot, snrm2, sasum         | sdotprod   | ✓           | ✓           |
| ddot, dnrm2, dasum         | ddotprod   | ✓           | ✓           |
| cdotu, cdotc               | cdotprod   | ✓           | tuning      |
| zdotu, zdotc               | zdotprod   | ✓           | tuning      |
| scnrm2, scasum             | scdotprod  | ✓           | tuning      |
| dznrm2, dzasum             | dzdotprod  | ✓           | tuning      |
| sdsdot                     | sdsdotprod | ✓           | ✓           |
| dsdot                      | dsdotprod  | ongoing     |             |
 

For example,

```shell
cd blas/dot/build
cmake ..
make demo_sdot_large_a10
```

will automatically synthesize this design into an image `blas/reconfigurable_dotprod/bin/sdotprod_large_a10.a`, and link the image into the demo application `blas/dot/bin/demo_sdot_large_a10`. Here `large_a10` refers to the large-sized configuration defined for A10 FPGA in [parameters.h](./parameters.h).

Alternatively, one can install the pre-synthesized bitstreams and demo applications following the general instructions.

Running a demo application will generate performance metrics.

## Metrics


