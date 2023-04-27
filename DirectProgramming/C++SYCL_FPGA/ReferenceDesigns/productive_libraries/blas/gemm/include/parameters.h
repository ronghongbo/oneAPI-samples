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
#ifndef GEMM_CONST_PARAMS_H
#define GEMM_CONST_PARAMS_H


// Inner loop bounds, which are static constant parameters of the design
#ifdef TINY // For verifying correctness only
/*    #if defined(T16161)  // Match William's code: 256x256 matrices, 16x16x1 array
        #define KKK         1
        #define JJJ         16
        #define III         16
        #define JJ          16
        #define II          16
        #define KK          256
    #elif defined(T881) // Match William's code: 8x8x1 array, each PE computes 1 value (On host: make I=8, J=8, K=64 to match their 64x64 matrices)
*/
        #define KKK         1
        #define JJJ         8
        #define III         8
        #define JJ          1
        #define II          1
        #define KK          1
/*
        #define KKK         4
        #define JJJ         4
        #define III         4
        #define JJ          4
        #define II          4
        #define KK          4
*/
/*    #elif defined(T441) // My mini version: 16x16 matrices 4x4x1 array
        #define KKK         1
        #define JJJ         4
        #define III         4
        #define JJ          4
        #define II          4
        #define KK          16
    #elif defined(T222) // My mini version: 4x4 matrices 2x2x2 array
        #define KKK         2
        #define JJJ         2
        #define III         2
        #define JJ          2
        #define II          2
        #define KK          2
    #elif defined(T221) // My mini version: 4x4 matrices 2x2x1 array
        #define KKK         1
        #define JJJ         2
        #define III         2
        #define JJ          2
        #define II          2
        #define KK          4
    #else
        #define KKK         4
        #define JJJ         4
        #define III         4
        #define JJ          4
        #define II          4
        #define KK          4
*/
#else // LARGE
    #ifdef GPU
        #define KKK         8
        #define JJJ         8
        #define III         32
        #define JJ          8
        #define II          2
        #define KK          1
    #elif defined(S10)
        #ifdef T2SP_SGEMM
            #define KKK         16
            #define JJJ         16
            #define III         14
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_DGEMM)
            #define KKK         8
            #define JJJ         4
            #define III         8
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_CGEMM)
            #define KKK         16
            #define JJJ         16
            #define III         14
            #define JJ          32
            #define II          32
            #define KK          32
        #else
            #define KKK         4
            #define JJJ         6
            #define III         4
            #define JJ          32
            #define II          32
            #define KK          32
        #endif
    #else   // TARGET == A10
        #ifdef T2SP_SGEMM
            #define KKK         16
            #define JJJ         8
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_DGEMM)
            #define KKK         8
            #define JJJ         4
            #define III         8
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_CGEMM)
            #define KKK         8
            #define JJJ         4
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #else
            #define KKK         4
            #define JJJ         4
            #define III         4
            #define JJ          32
            #define II          32
            #define KK          32        
        #endif
    #endif
#endif

#if defined(T2SP_SGEMM)
    #define ZERO 0
    #define CONST_TYPE float
    #define TTYPE Float(32)
    #define KERNEL "sgemm"
#elif defined(T2SP_DGEMM)
    #define ZERO 0
    #define CONST_TYPE double
    #define TTYPE Float(64) 
    #define KERNEL "dgemm"
#elif defined(T2SP_CGEMM)
    #define ZERO complex32_t(0.0f, 0.0f)
    #define CONST_TYPE complex32_t
    #define TTYPE Complex(32)
    #define KERNEL "cgemm"
#elif defined(T2SP_ZGEMM)
    #define ZERO complex64_t(0.0, 0.0)
    #define CONST_TYPE complex64_t
    #define TTYPE Complex(64) 
    #define KERNEL "zgemm"
#endif

#endif 
