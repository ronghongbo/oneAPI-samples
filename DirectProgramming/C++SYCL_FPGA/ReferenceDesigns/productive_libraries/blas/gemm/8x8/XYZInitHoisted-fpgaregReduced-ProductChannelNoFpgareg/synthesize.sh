#devcloud_login -b A10OAPI walltime=24:00:00 ./synthesize.sh
cd /home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/II1-JJ1-KK1-III8-JJJ8-KKK1-XYZInitHoisted-fpgaregReduced-ProductChannelNoFpgareg
icpx  -DA10 -DT2SP_SGEMM -DTINY -I./include -fsycl -fintelfpga -o hardware_demo.o -c hardware_demo.cpp
icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=intel_a10gx_pac:pac_a10 -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsclock=360MHz hardware_demo.o  -o gemm_fpga --verbose

#icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=Arria10 -fsycl-device-code-split=per_kernel -fsycl-link=early -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsclock=360MHz h    ardware_demo.o  -o gemm_fpga --verbose

