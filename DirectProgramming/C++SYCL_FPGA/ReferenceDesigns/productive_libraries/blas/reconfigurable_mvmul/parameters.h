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
#pragma once

// Inner loop bounds, which are static constant parameters of the design
#ifdef TINY // For verifying correctness only
    #define III         4
    #define II          4
    #define KK          4
#else
    #ifdef S10
        #ifdef T2SP_SMVMUL
            #define III         64
            #define II          32
            #define KK          32
            #define ZERO        0
            #define TTYPE       Float(32)
            #define CONST_TYPE  float
        #else
            #error currently only support float type
        #endif
    #elif defined(A10)
        #ifdef T2SP_SMVMUL
            #define III         32
            #define II          32
            #define KK          32
            #define ZERO        0
            #define TTYPE       Float(32)
            #define CONST_TYPE  float
        #else
            #error currently only support float type
        #endif
    #else
        #error No FPGA hardware platform (A10 or S10) specified
    #endif
#endif
