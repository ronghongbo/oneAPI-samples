#pragma once

#ifdef TINY
    #define KKK 4
    #ifdef T2SP_SVECADD
        #define TTYPE Float(32)
        #define CONST_TYPE float
    #elif defined(T2SP_DVECADD)
        #define TTYPE Float(64)
        #define CONST_TYPE double
    #elif defined(T2SP_CVECADD)
        #define TTYPE Complex(32)
        #define CONST_TYPE complex32_t
    #elif defined(T2SP_ZVECADD)
        #define TTYPE Complex(64)
        #define CONST_TYPE complex64_t
    #endif
#else
    #ifdef T2SP_SVECADD
        #define KKK 16
        #define TTYPE Float(32)
        #define CONST_TYPE float
    #elif defined(T2SP_DVECADD)
        #define KKK 8
        #define TTYPE Float(64)
        #define CONST_TYPE double
    #elif defined(T2SP_CVECADD)
        #define KKK 8
        #define TTYPE Complex(32)
        #define CONST_TYPE complex32_t
    #elif defined(T2SP_ZVECADD)
        #define KKK 4
        #define TTYPE Complex(64)
        #define CONST_TYPE complex64_t
    #endif
#endif
