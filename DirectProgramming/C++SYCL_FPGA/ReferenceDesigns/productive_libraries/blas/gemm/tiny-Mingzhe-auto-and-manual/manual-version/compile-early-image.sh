#icpx  -DA10 -DT2SP_SGEMM -DTINY -I./../../../../tools/Halide/include -I./include  -std=gnu++11 -o CMakeFiles/gemm_generate_tiny_sgemm.dir/gemm.o -c ./gemm.cpp
#icpx    -rdynamic CMakeFiles/gemm_generate_tiny_sgemm.dir/gemm.o  -o gemm_generate_tiny_sgemm  -L./../../../../tools/Halide/lib -Wl,-rpath,./../../../../tools/Halide/lib -lpthread -lz -ldl -lHalide 
icpx  -DA10 -DTINY -I../../../../tools/Halide/include -I. -I./../../../../../../include  -fsycl -fintelfpga -o gemm.o -c ./gemm.cpp
icpx  -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=/opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp:pac_a10 -Xsffp-reassociate -Xsprofile -Xsclock=360MHz -fsycl-link=early gemm.o  -o gemm_report.a  
