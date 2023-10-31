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
#include "util.h"

// Constant parameters (inner loop bounds) of the design
#include "parameters.h"

using namespace Halide;

int main()
{
    // Dependences
    #define P               iii,   ii, kk,      k,   i
    #define P_iii_minus_1   iii-1, ii, kk,      k,   i
    #define P_kk_minus_1    iii,   ii, kk-1,    k,   i
    #define P_k_minus_1     iii,   ii, kk+KK-1, k-1, i
    #define P_Out           iii,   ii,               i

    // Linearized addresses
    #define total_i         (iii + III * ii + III * II * i)
    #define total_k         (kk + KK * k)

    // Outer loop bounds, which are determined by input sizes
    #define MATRICES_I      (A.dim(1).extent())
    #define MATRICES_K      (B.dim(0).extent())
    #define I ((A.dim(1).extent() + (III * II - 1)) / (III * II))
    #define K ((A.dim(0).extent() + (KK - 1)) / KK)
    #define addr_A_in_range select(!Trans, total_i < MATRICES_I && total_k < MATRICES_K, total_k < MATRICES_I && total_i < MATRICES_K)

    // Inputs
    ImageParam A("A", TTYPE, 2), X("X", TTYPE, 1), Y("Y", TTYPE, 1);
    Param<CONST_TYPE> Alpha("Alpha"), Beta("Beta");
    Param<bool> Trans("Transpose");
    Param<bool> Conj("Conjugate");
    Param<int> IncX("IncX"), IncY("IncY");

    X.dim(0).set_stride(IncX);
    Y.dim(0).set_stride(IncY);


    Expr Check_A = select(addr_A_in_range, conditional_conjugate(Conj, A(select(!TransA, total_k, total_i), select(!TransA, total_i, total_k))), ZERO);
    Expr Check_X = select(total_i < MATRICES_I, X(P_Out), ZERO);
    Expr Check_Y = select(total_i < MATRICES_I, Y(P_Out), ZERO);

    // UREs
    Var kkk("kkk"), iii("iii"), kk("kk"), ii("ii"), k("k"), i("i");
    URE uA("uA", TTYPE, {P}), uX("uX", TTYPE, {P}), uZ("uZ", TTYPE, {P}), Product("Product");
    URE Add("Add", TTYPE, {P_Out}), Out("Out");
    uA(P) = A(total_k, total_i);
    uX(P) = select(iii == 0, X(total_k), uX(P_iii_minus_1));
    uZ(P) = select(kk == 0 && k == 0, ZERO,
                  select(kk == 0, uZ(P_k_minus_1), uZ(P_kk_minus_1))
                  ) + uA(P) * uX(P);
    Product(P_Out) = select(kk == KK-1 && k == K-1, uZ(P));

    Add(P_Out) = Alpha * Out(P_Out) + Beta * Y(P_Out);
    Out(P_Out) = select(true, Add(P_Out));

    // Put all the UREs inside the same loop nest of X.
    uA.merge_ures(uX, uZ, Out);
    Add.merge_ures(Out);

    // Explicitly set the loop bounds
    uA.set_bounds(iii, 0, III)
      .set_bounds(kk,  0, KK,  ii,  0, II)
      .set_bounds(k,   0, K,   i,   0, I);
    Add.set_bounds(iii, 0, III)
       .set_bounds(ii, 0, II)
       .set_bounds(i, 0, I)

    // Create a systolic array
    uA.space_time_transform(iii);
    Add.vectorize(iii);

    // I/O network
    Stensor DA("aLoader", DRAM);
    Stensor DX("xLoader", DRAM), SX("xFeeder", SRAM);
    Stensor DY("unloader", DRAM), Output("deserializer");
    Stensor RCollector("RCollector", REG), SCollector("SCollector", SRAM);
    Check_A >> DA.out(iii) >> FIFO(256);
    Check_X >> DX >> FIFO(256) >> SX.scope(k) >> FIFO(256);
    Product >> RCollector.scope(iii) >> FIFO(256) >> SCollector >> FIFO(256);
    Out >> FIFO(256) >> DY >> Output(total_i);

    Output.compile_to_onapi(OUTPUT_FILE, {Tran, Conj, Alpha, A, IncX, X, Beta, IncY, Y}, KERNEL, IntelFPGA);
    printf("Success\n");
    return 0;
}
