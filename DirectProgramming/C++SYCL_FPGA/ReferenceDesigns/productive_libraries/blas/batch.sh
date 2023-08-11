#!/bin/bash

# Usage:
#   ./batch.sh a10|s10

array = (dot     3 sdot   ddot  dsdot
         sdsdot  1 sdsdot
         dotc    2 cdotc  zdotc
         dotu    2 cdotu  zdotu
         nrm2    4 snrm2  dnrm2 scnrm2 dznrm2
         asum    4 sasum  dasum scasum dzasum
         axpy    4 saxpy  daxpy caxpy  zaxpy
         scal    4 sscal  dscal cscal  zscal
         copy    4 scopy  dcopy ccopy  zcopy
         gemm    4 sgemm  dgemm cgemm  zgemm
         symm    4 ssymm  dsymm csymm  zsymm
         hemm    2 chemm  zhemm)

index=0
while [ "$index" -lt "${#array[*]}" ]; do
    kernel=${array[$index]}
    echo "**** Building tests of " ${kernel}

    mkdir -p ${kernel}/build
    cd ${kernel}/build

    if [ "$1" = "a10" ]; then
        cmake ..
    else
        cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
    fi

    make tests
    ../bin/tests.sh

    num_variations=${array[$((index+1))]}
    let index=index+2
    for (( v=0; v<$num_variations; v++ )); do
        variation=${array[$((index))]}
        echo "**** Building demo of " ${variation}

        ../../install_pre_gen.sh ${variation}_large_$1
        make demo_${variation}_large_$1
        ../bin/demo_${variation}_large_$1

        let index=index+1
    done

    cd -
done