#devcloud_login -b A10OAPI walltime=24:00:00 ./synthesize.sh
source /glob/development-tools/versions/oneapi/2023.1.2/oneapi/setvars.sh
cd /home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/tiny-Mingzhe-auto-and-manual/manual-version-signalIJK

#icpx  -DA10 -DT2SP_SGEMM -DTINY -I./../../../../tools/Halide/include -I./include  -std=gnu++11 -o CMakeFiles/gemm_generate_tiny_sgemm.dir/gemm.o -c ./gemm.cpp
#icpx    -rdynamic CMakeFiles/gemm_generate_tiny_sgemm.dir/gemm.o  -o gemm_generate_tiny_sgemm  -L./../../../../tools/Halide/lib -Wl,-rpath,./../../../../tools/Halide/lib -lpthread -lz -ldl -lHalide 
 
icpx  -DA10 -DTINY -I../../../../tools/Halide/include -I. -I./../../../../../../include  -fsycl -fintelfpga -o gemm.o -c ./gemm.cpp
icpx  -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=/opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp:pac_a10 -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz gemm.o  -o gemm_fpga  

