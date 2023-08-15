#!/bin/bash
# This file records which kernels have pre-generated info. It is supposed to be included to another bash script only.

# A dictionary mapping from a kernel to the reconfigurable array that implements the kernel
declare -A kernel_to_reconfigurable_array
kernel_to_reconfigurable_array=(
    ["sgemm_large_a10"]="reconfigurable_matmul"
    ["ssymm_large_a10"]="reconfigurable_matmul"
    ["dgemm_large_a10"]="reconfigurable_matmul"
    ["sdot_large_a10"]="reconfigurable_dotprod"
    ["ddot_large_a10"]="reconfigurable_dotprod"
    ["dsdot_large_a10"]="reconfigurable_dotprod"
    ["sdsdot_large_a10"]="reconfigurable_dotprod"
    ["cdotu_large_a10"]="reconfigurable_dotprod"
    ["zdotu_large_a10"]="reconfigurable_dotprod"
    ["cdotc_large_a10"]="reconfigurable_dotprod"
    ["zdotc_large_a10"]="reconfigurable_dotprod"
    ["sasum_large_a10"]="reconfigurable_dotprod"
    ["dasum_large_a10"]="reconfigurable_dotprod"
    ["scasum_large_a10"]="reconfigurable_dotprod"
    ["dzasum_large_a10"]="reconfigurable_dotprod"
    ["snrm2_large_a10"]="reconfigurable_dotprod"
    ["dnrm2_large_a10"]="reconfigurable_dotprod"
    ["scnrm2_large_a10"]="reconfigurable_dotprod"
    ["dznrm2_large_a10"]="reconfigurable_dotprod"
    ["saxpy_large_a10"]="reconfigurable_vecadd"
    ["daxpy_large_a10"]="reconfigurable_vecadd"
    ["caxpy_large_a10"]="reconfigurable_vecadd"
    ["zaxpy_large_a10"]="reconfigurable_vecadd"
    ["sscal_large_a10"]="reconfigurable_vecadd"
    ["dscal_large_a10"]="reconfigurable_vecadd"
    ["cscal_large_a10"]="reconfigurable_vecadd"
    ["zscal_large_a10"]="reconfigurable_vecadd"
    ["scopy_large_a10"]="reconfigurable_vecadd"
    ["dcopy_large_a10"]="reconfigurable_vecadd"
    ["ccopy_large_a10"]="reconfigurable_vecadd"
    ["zcopy_large_a10"]="reconfigurable_vecadd"
    ["sgemm_large_s10"]="reconfigurable_matmul"
    ["dgemm_large_s10"]="reconfigurable_matmul"
    ["cgemm_tiny_s10"]="reconfigurable_matmul"
    ["ssymm_large_s10"]="reconfigurable_matmul"
    ["dsymm_large_s10"]="reconfigurable_matmul"
    ["csymm_tiny_s10"]="reconfigurable_matmul"
    ["chemm_tiny_s10"]="reconfigurable_matmul"
)
 
# A dictionary mapping from a kernel to a tarball of pre-generated source and bitstream.
declare -A kernel_to_tarball
kernel_to_tarball=(
    ["sgemm_large_a10"]="ssssmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["ssymm_large_a10"]="ssssmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dgemm_large_a10"]="ddddmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["sdot_large_a10"]="sdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["ddot_large_a10"]="ddotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dsdot_large_a10"]="dsdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["sdsdot_large_a10"]="sdsdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["cdotu_large_a10"]="cdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["zdotu_large_a10"]="zdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["cdotc_large_a10"]="cdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["zdotc_large_a10"]="zdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["sasum_large_a10"]="sdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dasum_large_a10"]="ddotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["scasum_large_a10"]="cdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dzasum_large_a10"]="zdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["snrm2_large_a10"]="sdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dnrm2_large_a10"]="ddotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["scnrm2_large_a10"]="cdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dznrm2_large_a10"]="zdotprod_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["saxpy_large_a10"]="svecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["daxpy_large_a10"]="dvecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["caxpy_large_a10"]="cvecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["zaxpy_large_a10"]="zvecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["sscal_large_a10"]="svecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dscal_large_a10"]="dvecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["cscal_large_a10"]="cvecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["zscal_large_a10"]="zvecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["scopy_large_a10"]="svecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dcopy_large_a10"]="dvecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["ccopy_large_a10"]="cvecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["zcopy_large_a10"]="zvecadd_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["sgemm_large_s10"]="ssssmatmul_large_s10_oneapi2023.2.tar.gz"
    ["dgemm_large_s10"]="ddddmatmul_large_s10_oneapi2023.2.tar.gz"
    ["cgemm_tiny_s10"]="ccccmatmul_tiny_s10_oneapi2023.2.tar.gz"
    ["ssymm_large_s10"]="ssssmatmul_large_s10_oneapi2023.2.tar.gz"
    ["dsymm_large_s10"]="ddddmatmul_large_s10_oneapi2023.2.tar.gz"
    ["csymm_tiny_s10"]="ccccmatmul_tiny_s10_oneapi2023.2.tar.gz"
    ["chemm_tiny_s10"]="ccccmatmul_tiny_s10_oneapi2023.2.tar.gz"
)

# A dictionary mapping from a kernel to the demo directory where a demo application is located
declare -A kernel_to_demo_dir
kernel_to_demo_dir=(
    ["sgemm_large_a10"]="gemm/bin"
    ["ssymm_large_a10"]="symm/bin"
    ["dgemm_large_a10"]="gemm/bin"
    ["sdot_large_a10"]="dot/bin"
    ["ddot_large_a10"]="dot/bin"
    ["dsdot_large_a10"]="dot/bin"
    ["sdsdot_large_a10"]="sdsdot/bin"
    ["cdotu_large_a10"]="dotu/bin"
    ["zdotu_large_a10"]="dotu/bin"
    ["cdotc_large_a10"]="dotc/bin"
    ["zdotc_large_a10"]="dotc/bin"
    ["sasum_large_a10"]="asum/bin"
    ["dasum_large_a10"]="asum/bin"
    ["scasum_large_a10"]="asum/bin"
    ["dzasum_large_a10"]="asum/bin"
    ["snrm2_large_a10"]="nrm2/bin"
    ["dnrm2_large_a10"]="nrm2/bin"
    ["scnrm2_large_a10"]="nrm2/bin"
    ["dznrm2_large_a10"]="nrm2/bin"
    ["saxpy_large_a10"]="axpy/bin"
    ["daxpy_large_a10"]="axpy/bin"
    ["caxpy_large_a10"]="axpy/bin"
    ["zaxpy_large_a10"]="axpy/bin"
    ["sscal_large_a10"]="scal/bin"
    ["dscal_large_a10"]="scal/bin"
    ["cscal_large_a10"]="scal/bin"
    ["zscal_large_a10"]="scal/bin"
    ["scopy_large_a10"]="copy/bin"
    ["dcopy_large_a10"]="copy/bin"
    ["ccopy_large_a10"]="copy/bin"
    ["zcopy_large_a10"]="copy/bin"
    ["sgemm_large_s10"]="gemm/bin"
    ["dgemm_large_s10"]="gemm/bin"
    ["cgemm_tiny_s10"]="gemm/bin"
    ["ssymm_large_s10"]="symm/bin"
    ["dsymm_large_s10"]="symm/bin"
    ["csymm_tiny_s10"]="symm/bin"
    ["chemm_tiny_s10"]="hemm/bin"
)
    
# A dictionary mapping from a kernel to a demo application
declare -A kernel_to_demo
kernel_to_demo=(
    ["sgemm_large_a10"]="demo_sgemm_large_a10.unsigned"
    ["ssymm_large_a10"]="demo_ssymm_large_a10.unsigned"
    ["dgemm_large_a10"]="demo_dgemm_large_a10.unsigned"
    ["sdot_large_a10"]="demo_sdot_large_a10.unsigned"
    ["ddot_large_a10"]="demo_ddot_large_a10.unsigned"
    ["dsdot_large_a10"]="demo_dsdot_large_a10.unsigned"
    ["sdsdot_large_a10"]="demo_sdsdot_large_a10.unsigned"
    ["cdotu_large_a10"]="demo_cdotu_large_a10.unsigned"
    ["zdotu_large_a10"]="demo_zdotu_large_a10.unsigned"
    ["cdotc_large_a10"]="demo_cdotc_large_a10.unsigned"
    ["zdotc_large_a10"]="demo_zdotc_large_a10.unsigned"
    ["sasum_large_a10"]="demo_sasum_large_a10.unsigned"
    ["dasum_large_a10"]="demo_dasum_large_a10.unsigned"
    ["scasum_large_a10"]="demo_scasum_large_a10.unsigned"
    ["dzasum_large_a10"]="demo_dzasum_large_a10.unsigned"
    ["snrm2_large_a10"]="demo_snrm2_large_a10.unsigned"
    ["dnrm2_large_a10"]="demo_dnrm2_large_a10.unsigned"
    ["scnrm2_large_a10"]="demo_scnrm2_large_a10.unsigned"
    ["dznrm2_large_a10"]="demo_dznrm2_large_a10.unsigned"
    ["saxpy_large_a10"]="demo_saxpy_large_a10.unsigned"
    ["daxpy_large_a10"]="demo_daxpy_large_a10.unsigned"
    ["caxpy_large_a10"]="demo_caxpy_large_a10.unsigned"
    ["zaxpy_large_a10"]="demo_zaxpy_large_a10.unsigned"
    ["sscal_large_a10"]="demo_sscal_large_a10.unsigned"
    ["dscal_large_a10"]="demo_dscal_large_a10.unsigned"
    ["cscal_large_a10"]="demo_cscal_large_a10.unsigned"
    ["zscal_large_a10"]="demo_zscal_large_a10.unsigned"
    ["scopy_large_a10"]="demo_scopy_large_a10.unsigned"
    ["dcopy_large_a10"]="demo_dcopy_large_a10.unsigned"
    ["ccopy_large_a10"]="demo_ccopy_large_a10.unsigned"
    ["zcopy_large_a10"]="demo_zcopy_large_a10.unsigned"
    ["sgemm_large_s10"]="demo_sgemm_large_s10"
    ["dgemm_large_s10"]="demo_dgemm_large_s10"
    ["cgemm_tiny_s10"]="demo_cgemm_tiny_s10"
    ["ssymm_large_s10"]="demo_ssymm_large_s10"
    ["dsymm_large_s10"]="demo_dsymm_large_s10"
    ["csymm_tiny_s10"]="demo_csymm_tiny_s10"
    ["chemm_tiny_s10"]="demo_chemm_tiny_s10"
)


