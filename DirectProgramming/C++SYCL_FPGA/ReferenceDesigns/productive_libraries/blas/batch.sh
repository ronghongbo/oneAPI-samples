#!/bin/bash

# Usage:
#   ./batch.sh a10|s10

array=(dot     3 sdot   ddot  dsdot
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

RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOLOR='\033[0m'

index=0
while [ "$index" -lt "${#array[*]}" ]; do
    kernel=${array[$index]}

    mkdir -p ${kernel}/build
    cd ${kernel}/build

    echo -e ${GREEN}Configuring ${kernel}${NOCOLOR}
    if [ "$1" = "a10" ]; then
        cmake .. >> ../../batch.out
    else
        cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10 >> ../../batch.out
    fi

    let original_index=index
    num_variations=${array[$((index+1))]}
    let index=index+2
    for (( v=0; v<$num_variations; v++ )); do
        variation=${array[$((index))]}
        echo -e ${GREEN}Cleaning ${variation}_tiny_$1 and ${variation}_large_$1${NOCOLOR}
        make clean_${variation}_tiny_$1 >> ../../batch.out
        make clean_${variation}_large_$1 >> ../../batch.out
        let index=index+1
    done
    let index=original_index

    echo -e ${GREEN}Building tests of ${kernel}${NOCOLOR}
    rm ../bin/test*
    make tests  >> ../../batch.out 2>&1

    echo -e ${GREEN}Running tests of ${kernel}${NOCOLOR}
    ../bin/tests.sh

    num_variations=${array[$((index+1))]}
    let index=index+2
    for (( v=0; v<$num_variations; v++ )); do
        variation=${array[$((index))]}

        echo -e ${GREEN}Installing pre-generated files for ${variation}_large_$1${NOCOLOR}
        if ../../install_pre_gen.sh ${variation}_large_$1 >> ../../batch.out; then
            echo -e ${GREEN}Making demo of ${variation}_large_$1${NOCOLOR}
            make demo_${variation}_large_$1 >> ../../batch.out

            echo -e ${GREEN}Running demo of ${variation}_large_$1${NOCOLOR}
            if [ "$1" = "a10" ]; then
                ../bin/demo_${variation}_large_$1.unsigned
            else
                ../bin/demo_${variation}_large_$1
            fi
        else
            echo -e Sorry, it seems no pre-generated files exist for ${variation}_large_$1. Skip building the demo due to the long FPGA synthesis time.
        fi
        let index=index+1
    done

    cd -
done
