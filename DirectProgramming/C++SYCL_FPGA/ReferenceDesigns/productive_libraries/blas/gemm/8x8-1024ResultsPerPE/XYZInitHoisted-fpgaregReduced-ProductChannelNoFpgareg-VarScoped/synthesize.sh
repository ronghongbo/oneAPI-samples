#devcloud_login -b A10OAPI walltime=24:00:00 ./synthesize.sh
cd /home/u89062/original-oneAPI-samples/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/8x8-1024ResultsPerPE/XYZInitHoisted-fpgaregReduced-ProductChannelNoFpgareg-VarScoped
icpx  -DA10 -DT2SP_SGEMM -I./include -fsycl -fintelfpga -o hardware_demo.o -c hardware_demo.cpp
icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=/opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp:pac_a10 -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsclock=360MHz hardware_demo.o  -o gemm_fpga --verbose

#icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=Arria10 -fsycl-device-code-split=per_kernel -fsycl-link=early -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsclock=360MHz h    ardware_demo.o  -o gemm_fpga --verbose

