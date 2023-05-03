#devcloud_login -b A10OAPI walltime=24:00:00 ./synthesize.sh
set -x
cd /home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/II32-JJ32-KK32-III10-JJJ8-KKK16-XYZInitHoisted-fpgaregReduced-ProductChannelNoFpgareg
icpx  -DA10 -DT2SP_SGEMM -I. -fsycl -fintelfpga -o hardware_demo.o -c hardware_demo.cpp
icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=intel_a10gx_pac:pac_a10 -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile hardware_demo.o  -o gemm_fpga --verbose

