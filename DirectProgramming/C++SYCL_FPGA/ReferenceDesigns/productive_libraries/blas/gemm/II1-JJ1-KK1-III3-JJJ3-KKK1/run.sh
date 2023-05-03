#devcloud_login -b A10OAPI walltime=24:00:00 ./run.sh
set -x
#cd /home/u89062/gemm-synthesis-issue/II8-JJ8-KK8-III8-JJJ8-KKK8
icpx  -DA10 -DT2SP_SGEMM -DTINY -I. -fsycl -fintelfpga -o hardware_demo.o -c hardware_demo.cpp
icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=Arria10 -fsycl-device-code-split=per_kernel -fsycl-link=early -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsclock=360MHz hardware_demo.o  -o gemm_fpga --verbose

