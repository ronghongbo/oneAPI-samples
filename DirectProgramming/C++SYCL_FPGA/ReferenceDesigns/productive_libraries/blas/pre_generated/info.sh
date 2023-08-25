#!/bin/bash
# This file records which kernels have pre-generated info. It is supposed to be included to another bash script only.

# A dictionary mapping from a kernel to the reconfigurable array that implements the kernel
declare -A kernel_to_reconfigurable_array
kernel_to_reconfigurable_array=(
    # A10
    ["sgemm_large_a10"]="reconfigurable_matmul"
    ["dgemm_large_a10"]="reconfigurable_matmul"
    ["cgemm_large_a10"]="reconfigurable_matmul"
    ["zgemm_large_a10"]="reconfigurable_matmul"
    ["ssymm_large_a10"]="reconfigurable_matmul"
    ["dsymm_large_a10"]="reconfigurable_matmul"
    ["csymm_large_a10"]="reconfigurable_matmul"
    ["zsymm_large_a10"]="reconfigurable_matmul"
    ["chemm_large_a10"]="reconfigurable_matmul"
    ["zhemm_large_a10"]="reconfigurable_matmul"
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

    # S10
    ["sgemm_large_s10"]="reconfigurable_matmul"
    ["dgemm_large_s10"]="reconfigurable_matmul"
    ["cgemm_large_s10"]="reconfigurable_matmul"
    ["zgemm_large_s10"]="reconfigurable_matmul"
    ["ssymm_large_s10"]="reconfigurable_matmul"
    ["dsymm_large_s10"]="reconfigurable_matmul"
    ["csymm_large_s10"]="reconfigurable_matmul"
    ["zsymm_large_s10"]="reconfigurable_matmul"
    ["chemm_large_s10"]="reconfigurable_matmul"
    ["zhemm_large_s10"]="reconfigurable_matmul"
    ["sdot_large_s10"]="reconfigurable_dotprod"
    ["ddot_large_s10"]="reconfigurable_dotprod"
    ["dsdot_large_s10"]="reconfigurable_dotprod"
    ["sdsdot_large_s10"]="reconfigurable_dotprod"
    ["cdotu_large_s10"]="reconfigurable_dotprod"
    ["zdotu_large_s10"]="reconfigurable_dotprod"
    ["cdotc_large_s10"]="reconfigurable_dotprod"
    ["zdotc_large_s10"]="reconfigurable_dotprod"
    ["sasum_large_s10"]="reconfigurable_dotprod"
    ["dasum_large_s10"]="reconfigurable_dotprod"
    ["scasum_large_s10"]="reconfigurable_dotprod"
    ["dzasum_large_s10"]="reconfigurable_dotprod"
    ["snrm2_large_s10"]="reconfigurable_dotprod"
    ["dnrm2_large_s10"]="reconfigurable_dotprod"
    ["scnrm2_large_s10"]="reconfigurable_dotprod"
    ["dznrm2_large_s10"]="reconfigurable_dotprod"
    ["saxpy_large_s10"]="reconfigurable_vecadd"
    ["daxpy_large_s10"]="reconfigurable_vecadd"
    ["caxpy_large_s10"]="reconfigurable_vecadd"
    ["zaxpy_large_s10"]="reconfigurable_vecadd"
    ["sscal_large_s10"]="reconfigurable_vecadd"
    ["dscal_large_s10"]="reconfigurable_vecadd"
    ["cscal_large_s10"]="reconfigurable_vecadd"
    ["zscal_large_s10"]="reconfigurable_vecadd"
    ["scopy_large_s10"]="reconfigurable_vecadd"
    ["dcopy_large_s10"]="reconfigurable_vecadd"
    ["ccopy_large_s10"]="reconfigurable_vecadd"
    ["zcopy_large_s10"]="reconfigurable_vecadd"
)

# A dictionary mapping from a kernel to a tarball of pre-generated source and bitstream.
declare -A kernel_to_tarball
kernel_to_tarball=(
    # A10
    ["sgemm_large_a10"]="ssssmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dgemm_large_a10"]="ddddmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["cgemm_large_a10"]="ccccmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["zgemm_large_a10"]="zzzzmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["ssymm_large_a10"]="ssssmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["dsymm_large_a10"]="ddddmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["csymm_large_a10"]="ccccmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["zsymm_large_a10"]="zzzzmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["chemm_large_a10"]="ccccmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["zhemm_large_a10"]="zzzzmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
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

    # S10
    ["sgemm_large_s10"]="ssssmatmul_large_s10_oneapi2023.2.tar.gz"
    ["dgemm_large_s10"]="ddddmatmul_large_s10_oneapi2023.2.tar.gz"
    ["cgemm_large_s10"]="ccccmatmul_large_s10_oneapi2023.2.tar.gz"
    ["zgemm_large_s10"]="zzzzmatmul_large_s10_oneapi2023.2.tar.gz"
    ["ssymm_large_s10"]="ssssmatmul_large_s10_oneapi2023.2.tar.gz"
    ["dsymm_large_s10"]="ddddmatmul_large_s10_oneapi2023.2.tar.gz"
    ["csymm_large_s10"]="ccccmatmul_large_s10_oneapi2023.2.tar.gz"
    ["zsymm_large_s10"]="zzzzmatmul_large_s10_oneapi2023.2.tar.gz"
    ["chemm_large_s10"]="ccccmatmul_large_s10_oneapi2023.2.tar.gz"
    ["zhemm_large_s10"]="zzzzmatmul_large_s10_oneapi2023.2.tar.gz"
    ["sdot_large_s10"]="sdotprod_large_s10_oneapi2023.2.tar.gz"
    ["ddot_large_s10"]="ddotprod_large_s10_oneapi2023.2.tar.gz"
    ["dsdot_large_s10"]="dsdotprod_large_s10_oneapi2023.2.tar.gz"
    ["sdsdot_large_s10"]="sdsdotprod_large_s10_oneapi2023.2.tar.gz"
    ["cdotu_large_s10"]="cdotprod_large_s10_oneapi2023.2.tar.gz"
    ["zdotu_large_s10"]="zdotprod_large_s10_oneapi2023.2.tar.gz"
    ["cdotc_large_s10"]="cdotprod_large_s10_oneapi2023.2.tar.gz"
    ["zdotc_large_s10"]="zdotprod_large_s10_oneapi2023.2.tar.gz"
    ["sasum_large_s10"]="sdotprod_large_s10_oneapi2023.2.tar.gz"
    ["dasum_large_s10"]="ddotprod_large_s10_oneapi2023.2.tar.gz"
    ["scasum_large_s10"]="cdotprod_large_s10_oneapi2023.2.tar.gz"
    ["dzasum_large_s10"]="zdotprod_large_s10_oneapi2023.2.tar.gz"
    ["snrm2_large_s10"]="sdotprod_large_s10_oneapi2023.2.tar.gz"
    ["dnrm2_large_s10"]="ddotprod_large_s10_oneapi2023.2.tar.gz"
    ["scnrm2_large_s10"]="cdotprod_large_s10_oneapi2023.2.tar.gz"
    ["dznrm2_large_s10"]="zdotprod_large_s10_oneapi2023.2.tar.gz"
    ["saxpy_large_s10"]="svecadd_large_s10_oneapi2023.2.tar.gz"
    ["daxpy_large_s10"]="dvecadd_large_s10_oneapi2023.2.tar.gz"
    ["caxpy_large_s10"]="cvecadd_large_s10_oneapi2023.2.tar.gz"
    ["zaxpy_large_s10"]="zvecadd_large_s10_oneapi2023.2.tar.gz"
    ["sscal_large_s10"]="svecadd_large_s10_oneapi2023.2.tar.gz"
    ["dscal_large_s10"]="dvecadd_large_s10_oneapi2023.2.tar.gz"
    ["cscal_large_s10"]="cvecadd_large_s10_oneapi2023.2.tar.gz"
    ["zscal_large_s10"]="zvecadd_large_s10_oneapi2023.2.tar.gz"
    ["scopy_large_s10"]="svecadd_large_s10_oneapi2023.2.tar.gz"
    ["dcopy_large_s10"]="dvecadd_large_s10_oneapi2023.2.tar.gz"
    ["ccopy_large_s10"]="cvecadd_large_s10_oneapi2023.2.tar.gz"
    ["zcopy_large_s10"]="zvecadd_large_s10_oneapi2023.2.tar.gz"
)

