export AOCL_BOARD_PACKAGE_ROOT=/opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp
#/glob/development-tools/versions/fpgasupportstack/a10/1.2.1/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp
$INTELFPGAOCLSDKROOT/host/linux64/bin/aocl-extract-aocx -i gemm_fpga -o gemm.aocx
source $AOCL_BOARD_PACKAGE_ROOT/linux64/libexec/sign_aocx.sh -H openssl_manager -i gemm.aocx -r NULL -k NULL -o gemm_unsigned.aocx

icpx    -rdynamic -fsycl -fintelfpga -fsycl-add-targets=spir64_fpga-unknown-unknown:gemm_unsigned.aocx hardware_demo.o  -o gemm_fpga -lOpenCL -lsycl

