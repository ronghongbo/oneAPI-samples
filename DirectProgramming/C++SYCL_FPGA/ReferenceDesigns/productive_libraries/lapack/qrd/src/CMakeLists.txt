###############################################################################
### From the T2S specification of the QRD kernel, generate SYCL code for the kernel
###############################################################################
set(T2S_SPEC              qrd-spec.cpp)
set(T2S_SPEC_CONST_PARAMS "-DI=128 -DJ=128 -DK=128 -DTYPE=Float(32)")
set(SYCL_KERNEL_FILE      qrd.sycl.h)

add_custom_command(OUTPUT ${SYCL_KERNEL_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} ${T2S_SPEC} ${T2S_SPEC_CONST_PARAMS} -I${T2S_PATH}/Halide/include ${T2S_PATH}/Halide/lib/libHalide.a -lz -lpthread -ldl -std=c++11
    COMMAND env CLEARCODE=1 ./a.out
)

###############################################################################
### Generate bitstream from the SYCL code of the kernel, and test it.
#TODO: Enable separate compilation.
###############################################################################
