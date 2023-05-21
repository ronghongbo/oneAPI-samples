#devcloud_login -b A10OAPI walltime=24:00:00 ./synthesize.sh
source /glob/development-tools/versions/oneapi/2023.1.2/oneapi/setvars.sh
cd /home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/8x8-1024ResultsPerPE/signal-generator
set -x
SYCL_LINK=image #early

#icpx  -DA10 -DT2SP_SGEMM -DTINY -I/home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/../../tools/Halide/include -I.  -I/home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/../../../../include  -std=gnu++11 -o gemm.o -c gemm.cpp

#icpx  -rdynamic gemm.o  -o gemm_generate_tiny_sgemm  -L/home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/../../tools/Halide/lib -Wl,-rpath,/home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/../../tools/Halide/lib -lpthread -lz -ldl -lHalide 

#env CLEARCODE=1 ./gemm_generate_tiny_sgemm

icpx  -DA10 -DT2SP_SGEMM -DTINY -I/home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/../../tools/Halide/include -I.  -I/home/u89062/oneAPI-samples-ronghongbo/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/gemm/../../../../include  -fsycl -fintelfpga -o  hardware_demo.o -c hardware_demo.cpp

icpx    -rdynamic -fsycl -fintelfpga -reuse-exe=gemm_fpga -Xshardware -Xstarget=/glob/development-tools/versions/fpgasupportstack/a10/1.2.1/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp:pac_a10 -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz hardware_demo.o  -o gemm_fpga


#icpx -DA10 -DT2SP_SGEMM -DTINY -I../../../../tools/Halide/include -I../../include  -std=gnu++11 -o gemm.o -c ../../gemm.cpp
#icpx -rdynamic gemm.o  -o gemm_generate_tiny_sgemm  -L../../../../tools/Halide/lib -Wl,-rpath,../../../../tools/Halide/lib -lpthread -lz -ldl -lHalide
#env CLEARCODE=1 ./gemm_generate_tiny_sgemm 

# Test
#icpx -DFPGA_EMULATOR -DT2SP_NDEBUG -DMKL_ILP64 -DT2SP_TEST_0 -O0 -g -DTINY -I../../../../tools/Halide/include -I../../../../../../include -I../../include/ -I. -I/glob/development-tools/versions/oneapi/2023.1.2/oneapi/mkl/2023.1.0/include -fsycl -fintelfpga -o test.o -c ./test.cpp
#icpx -rdynamic -fsycl -fintelfpga test.o -O0 -g  -o test_0 -L../../../../tools/Halide/lib  -L/glob/development-tools/versions/oneapi/2023.1.2/oneapi/mkl/2023.1.0/lib/intel64 -Wl,-rpath,../../../../tools/Halide/lib:/glob/development-tools/versions/oneapi/2023.1.2/oneapi/mkl/2023.1.0/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl 

#icpx -DA10 -DT2SP_SGEMM -DTINY -I../../../../tools/Halide/include -I../../../../../../include -I../../include/ -I. -fsycl -fintelfpga -o hardware_demo.o -c ./hardware_demo.cpp
#icpx -rdynamic -fsycl -fintelfpga -fsycl-link=$SYCL_LINK -Xshardware -Xstarget=/opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp:pac_a10 -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz hardware_demo.o  -o hardware_demo

