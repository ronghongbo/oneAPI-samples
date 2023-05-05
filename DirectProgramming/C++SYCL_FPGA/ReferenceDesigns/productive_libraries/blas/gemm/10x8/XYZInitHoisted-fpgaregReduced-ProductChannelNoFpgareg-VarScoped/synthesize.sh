#devcloud_login -b A10OAPI walltime=24:00:00 ./synthesize.sh

# This does not work:
# Path to this script
#PATH_TO_SCRIPT="$( cd "$(dirname "$BASH_SOURCE" )" >/dev/null 2>&1 ; pwd -P )"
#echo Entering $PATH_TO_SCRIPT ....
#cd $PATH_TO_SCRIPT
cd $HOME/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/10x8/XYZInitHoisted-fpgaregReduced-ProductChannelNoFpgareg-VarScoped

icpx  -DA10 -DT2SP_SGEMM -I./include -I../../../../../include/  -fsycl -fintelfpga -o hardware_demo.o -c hardware_demo.cpp
#icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=intel_a10gx_pac:pac_a10 -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsclock=360MHz hardware_demo.o  -o gemm_fpga --verbose
icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=Arria10 -fsycl-device-code-split=per_kernel -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsclock=360MHz hardware_demo.o  -o gemm_fpga --verbose

