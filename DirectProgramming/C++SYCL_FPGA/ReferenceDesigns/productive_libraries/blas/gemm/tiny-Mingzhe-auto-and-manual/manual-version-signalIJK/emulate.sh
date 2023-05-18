#icpx  -DA10 -DT2SP_SGEMM -DTINY -I./../../../../tools/Halide/include -I./include  -std=gnu++11 -o CMakeFiles/gemm_generate_tiny_sgemm.dir/gemm.o -c ./gemm.cpp
#icpx    -rdynamic CMakeFiles/gemm_generate_tiny_sgemm.dir/gemm.o  -o gemm_generate_tiny_sgemm  -L./../../../../tools/Halide/lib -Wl,-rpath,./../../../../tools/Halide/lib -lpthread -lz -ldl -lHalide 
icpx  -DA10 -DTINY -DFPGA_EMULATOR -I../../../../tools/Halide/include -I. -I./../../../../../../include  -fsycl -fintelfpga -o gemm.o -c ./gemm.cpp
icpx  -rdynamic -fsycl -fintelfpga gemm.o  -o gemm_emu
