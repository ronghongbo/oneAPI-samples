#ifndef MATMUL_CONST_PARAMS_H
#define MATMUL_CONST_PARAMS_H

// Inner loop bounds, which are static constant parameters of the design
#ifdef TINY // For verifying correctness only
    #define KKK         4
    #define JJJ         4
    #define III         4
    #define JJ          4
    #define II          4
    #define KK          4
#else // LARGE
    #if defined(S10)
        #ifdef T2SP_SMATMUL
            #define KKK         16
            #define JJJ         16
            #define III         14
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_DMATMUL)
            #define KKK         8
            #define JJJ         4
            #define III         8
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_CMATMUL)
            #define KKK         16
            #define JJJ         16
            #define III         14
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_ZMATMUL)
            #define KKK         4
            #define JJJ         6
            #define III         4
            #define JJ          32
            #define II          32
            #define KK          32
        #else
            #error Precision is undefined. Define one of the following macros: T2SP_SMATMUL, T2SP_DMATMUL, T2SP_CMATMUL, and T2SP_ZMATMUL.
        #endif
    #elif defined(A10)
        #ifdef T2SP_SMATMUL
            #define KKK         16
            #define JJJ         8
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_DMATMUL)
            #define KKK         8
            #define JJJ         4
            #define III         8
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_CMATMUL)
            #define KKK         8
            #define JJJ         4
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #elif defined(T2SP_ZMATMUL)
            #define KKK         4
            #define JJJ         4
            #define III         4
            #define JJ          32
            #define II          32
            #define KK          32        
        #else
            #error Precision is undefined. Define one of the following macros: T2SP_SMATMUL, T2SP_DMATMUL, T2SP_CMATMUL, and T2SP_ZMATMUL.
        #endif
    #else
        #error The size of the systolic array is undefined. Define a precision (T2SP_SMATMUL, T2SP_DMATMUL, T2SP_CMATMUL, or T2SP_ZMATMUL). Then define hardware (A10 or S10), or define TINY, which indidates a tiny systolic array regardless of hardware.
    #endif
#endif

#if defined(T2SP_SMATMUL)
    #define ZERO       0
    #define CONST_TYPE float
    #define TTYPE      Float(32)
    #define KERNEL     "smatmul"
#elif defined(T2SP_DMATMUL)
    #define ZERO       0
    #define CONST_TYPE double
    #define TTYPE      Float(64)
    #define KERNEL     "dmatmul"
#elif defined(T2SP_CMATMUL)
    #define ZERO       complex32_t(0.0f, 0.0f)
    #define CONST_TYPE complex32_t
    #define TTYPE      Complex(32)
    #define KERNEL     "cmatmul"
#elif defined(T2SP_ZMATMUL)
    #define ZERO       complex64_t(0.0, 0.0)
    #define CONST_TYPE complex64_t
    #define TTYPE      Complex(64)
    #define KERNEL     "zmatmul"
#else
    #error Precision is undefined. Define one of the following macros: T2SP_SMATMUL, T2SP_DMATMUL, T2SP_CMATMUL, and T2SP_ZMATMUL.
#endif

#endif 
