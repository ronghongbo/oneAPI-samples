#!/bin/bash
# Usage: ./install_pre_gen.sh VARIATION_SIZE_HW
# Here VARIATION is a kernel's variation, SIZE is tiny or large, and HW is a10 or s10.

# Bash version must be >= 4
bash_version=$BASH_VERSINFO
echo Bash version: $bash_version
if (($bash_version<4)); then
    echo "Error: Bash version >= 4.0 expected to run the script"
    exit
fi

path_to_blas="$( cd "$(dirname $(realpath "$BASH_SOURCE") )" >/dev/null 2>&1 ; pwd -P )" # The path to this script, which is the blas directory
echo Entering productive BLAS: $path_to_blas
cd $path_to_blas

# A dictionary mapping from a kernel to a tarball of pre-generated source and bitstream. A tarball was created in this way: "cd blas; tar cvzf pre_generated/NAME.tar.gz reconfigurable_ARRAY/bin/IMAGE_FILE reconfigurable_ARRAY/oneapi/SYCL_FILE reconfigurable_ARRAY/reports/REPORTS_DIR". Here the capital words should be replaced appropriately. 
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

if test "${kernel_to_tarball[$1]+exists}"; then
    tarball=${kernel_to_tarball[$1]}
    echo Expanding $tarball ...
    tar xzvf pre_generated/$tarball --overwrite --touch
    # Somehow, the --touch above seems to work for directories, but not for files under the directories. To be sure, touch files manually
    for file in $(tar -tf pre_generated/$tarball 2>/dev/null)
    do
        touch $file
    done
else
    echo "Sorry. No pre-generated tarball for $1"
    exit 1
fi

if test "${kernel_to_demo[$1]+exists}"; then
    demo=${kernel_to_demo[$1]}
    demo_dir=${kernel_to_demo_dir[$1]}
    echo Recovering $demo to directory ${demo_dir} ...
    cp pre_generated/${demo} ${demo_dir}
    touch ${demo_dir}/${demo}
else
    echo "Sorry. No pre-generated demo for $1"
    exit 1
fi

cd -
