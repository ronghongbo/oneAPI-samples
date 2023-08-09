#pragma once

#ifdef TINY // For verifying correctness only
    #define KKK         4
    #define KK          4
#else
    #ifdef T2SP_DDOTPROD
        #define KK 64
        #define KKK 8
    #elif defined(T2SP_CDOTPROD)
        #define KK 64
        #define KKK 8
    #elif defined(T2SP_ZDOTPROD)
        #define KK 32
        #define KKK 4
    #elif defined(T2SP_SDSDOTPROD)
        #define KK 64
        #define KKK 8
    #elif defined(T2SP_SCDOTPROD)
        #define KK 64
        #define KKK 8
    #elif defined(T2SP_DZDOTPROD)
        #define KK 32
        #define KKK 4
    #elif defined(T2SP_DSDOTPROD)
        #define KK 64
        #define KKK 16
    #else
        #define KK 64
        #define KKK 16
    #endif
#endif

#ifdef T2SP_DDOTPROD
    #define ITYPE TTYPE
    #define TTYPE Float(64)
#elif defined(T2SP_CDOTPROD)
    #define ITYPE TTYPE
    #define TTYPE Complex(32)
#elif defined(T2SP_ZDOTPROD)
    #define ITYPE TTYPE
    #define TTYPE Complex(64)
#elif defined(T2SP_SDSDOTPROD)
    #define ITYPE Float(32)
    #define TTYPE Float(64)
#elif defined(T2SP_SCDOTPROD)
    #define ITYPE Complex(32)
    #define TTYPE Float(32)
#elif defined(T2SP_DZDOTPROD)
    #define ITYPE Complex(64)
    #define TTYPE Float(64)
#elif defined(T2SP_DSDOTPROD)
    #define ITYPE Float(32)
    #define TTYPE Float(64)
#else
    #define ITYPE TTYPE
    #define TTYPE Float(32)
#endif
