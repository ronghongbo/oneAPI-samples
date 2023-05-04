#devcloud_login -b A10OAPI walltime=24:00:00 ~/gemm-synthesis-issue/large-sgemm-clearOneapi/synthesize-gemm-fpga-repro-oneapi.sh
set -x
cd /home/u89062/gemm-synthesis-issue/large-sgemm-clearOneapi
icpx  -DA10 -DT2SP_SGEMM -I. -fsycl -fintelfpga -o hardware_demo.o -c hardware_demo.cpp
icpx -rdynamic -fsycl -fintelfpga -Xshardware -Xstarget=intel_a10gx_pac:pac_a10 -Xsffp-reassociate -Xsffp-contract=fast -Xsprofile -Xsseed=1 hardware_demo.o  -o gemm_fpga --verbose

