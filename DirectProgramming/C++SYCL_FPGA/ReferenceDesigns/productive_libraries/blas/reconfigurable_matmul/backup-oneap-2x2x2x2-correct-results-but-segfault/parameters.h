#ifndef MATMUL_CONST_PARAMS_H
#define MATMUL_CONST_PARAMS_H

// NOTE: When change any parameters of the systolic array (KKK, JJJ, III, JJ, II, KK), please make the same change in api.hpp

// Inner loop bounds, which are static constant parameters of the design
#ifdef TINY // For verifying correctness only
    #define KKK         2
    #define JJJ         2
    #define III         2
    #define JJ          2
    #define II          2
    #define KK          2
#else // LARGE
    #if defined(S10)
        #ifdef TYPEC_S
            #define KKK         16
            #define JJJ         16
            #define III         14
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_D
            #define KKK         8
            #define JJJ         4
            #define III         8
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_C
            #define KKK         16
            #define JJJ         16
            #define III         14
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_Z
            #define KKK         4
            #define JJJ         6
            #define III         4
            #define JJ          32
            #define II          32
            #define KK          32
        #else
            #error Precision of the output matrix is undefined.
        #endif
    #elif defined(A10)
        #ifdef TYPEC_S
            #define KKK         16
            #define JJJ         8
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_D
            #define KKK         8
            #define JJJ         4
            #define III         8
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_C
            #define KKK         8
            #define JJJ         4
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_Z
            #define KKK         4
            #define JJJ         4
            #define III         4
            #define JJ          32
            #define II          32
            #define KK          32
        #else
            #error Precision of the output matrix is undefined.
        #endif
    #else
        #error The size of the systolic array is undefined. Define the precision of the output matrix. Then define hardware (A10 or S10), or define TINY, which indidates a tiny systolic array regardless of hardware.
    #endif
#endif

#if TYPEC_S
    #define ZERO        0
    #define SCALAR_ZERO 0
#elif TYPEC_D
    #define ZERO       0
#elif TYPEC_C
    #define ZERO       complex32_t(0.0f, 0.0f)
#elif TYPEC_Z
    #define ZERO       complex64_t(0.0, 0.0)
#else
    #error Precision of the output matrix is undefined.
#endif

#if TYPE_SCALAR_S
    #define SCALAR_ZERO 0
#elif TYPE_SCALAR_D
    #define SCALAR_ZERO 0
#elif TYPE_SCALAR_C
    #define SCALAR_ZERO complex32_t(0.0f, 0.0f)
#elif TYPE_SCALAR_Z
    #define SCALAR_ZERO complex64_t(0.0, 0.0)
#else
    #error Precision of beta and alpha is undefined.
#endif

#endif