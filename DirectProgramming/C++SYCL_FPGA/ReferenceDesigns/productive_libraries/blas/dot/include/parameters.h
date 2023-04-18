#pragma once

#define KK  64

#ifdef TINY
    #define KKK 4
#elif defined(S10)
    #ifdef T2SP_SDOT
        #define KKK 32
    #elif defined(T2SP_DDOT)
        #define KKK 16
    #endif
#elif defined(A10)
    #ifdef T2SP_SDOT
        #define KKK 16
    #elif defined(T2SP_DDOT)
        #define KKK 8
    #endif
#endif

#ifdef T2SP_SDOT
    #define TTYPE Float(32)
    #define KERNEL "sdot"
#elif defined(T2SP_DDOT)
    #define TTYPE Float(64)
    #define KERNEL "ddot"
#endif
