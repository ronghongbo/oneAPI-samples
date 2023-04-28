icpx  -DA10 -DT2SP_SGEMM -DTINY -I./include -fsycl -fintelfpga -o hardware_demo.o -c hardware_demo.cpp
icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=Arria10 -fsycl-device-code-split=per_kernel -fsycl-link=early -Xsffp-reassociate -Xsprofile -Xsclock=360MHz hardware_demo.o  -o gemm_fpga --verbose


