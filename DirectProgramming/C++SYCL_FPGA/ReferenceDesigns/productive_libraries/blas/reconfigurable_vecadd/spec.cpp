/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the BSD-2-Clause Plus Patent License (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://opensource.org/licenses/BSDplusPatent
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: BSD-2-Clause-Patent
*******************************************************************************/
#include "Halide.h"
#include "parameters.h"

using namespace Halide;

int main()
{
    // Dependences
    #define P                kkk, k,     b
    // Linearized addresses
    #define total_k         (kkk + KKK * k)

    // Outer loop bounds, which are determined by input sizes
    #define K ((X.dim(0).extent() + KKK - 1) / KKK)
    #define B (X.dim(1).extent())

    #define addr_in_range (KKK * k < X.dim(0).extent())

    // Inputs
    ImageParam X("X", TTYPE, 2);
    ImageParam Y("Y", TTYPE, 2);
    Param<int> IncX("IncX");
    Param<int> IncY("IncY");
    Param<CONST_TYPE> Alpha("Alpha"), Beta("Beta"); 

    X.dim(0).set_stride(IncX);
    Y.dim(0).set_stride(IncY);

    // UREs
    Var kkk("kkk"), kk("kk"), k("k"), b("b");
    URE uY("uY", TTYPE, {P}), uX("uX", TTYPE, {P}), uZ_1("uZ_1", TTYPE, {P}), Z("Z");

    Expr Check_Load_X = select(addr_in_range, X(total_k, b), 0);
    Expr Check_Load_Y = select(addr_in_range, Y(total_k, b), 0);

    uX(P) = Check_Load_X;
    uY(P) = Check_Load_Y;
    uZ_1(P) = Alpha * uX(P) + Beta * uY(P);
    Z(P) = select(true, uZ_1(P));

    // Put all the UREs inside the same loop nest of X.
    uX.merge_ures(uY, uZ_1, Z);

    // Explicitly set the loop bounds
    uX.set_bounds(kkk,  0, KKK, k,  0, K)
      .set_bounds(b,    0, B);
    uX.vectorize(kkk);

    // I/O network
    Stensor DX("xLoader", DRAM), DY("yLoader", DRAM), DC("unloader", DRAM), C("deserializer");
    Check_Load_X >> DX >> FIFO(256);
    Check_Load_Y >> DY >> FIFO(256);
    Z >> FIFO(256) >> DC >> C(b);

    // Compile the kernel to an FPGA bitstream, and expose a C interface for the host to invoke
    C.compile_to_oneapi(OUTPUT_FILE, {Alpha, X, IncX, Beta, Y, IncY}, KERNEL, IntelFPGA);
    printf("Success\n");
    return 0;
}
