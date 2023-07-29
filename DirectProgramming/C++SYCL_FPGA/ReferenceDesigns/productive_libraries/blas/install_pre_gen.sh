#!/bin/bash 
# Usage: ./install_pre_gen.sh kernel_size_hardware

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

# A dictionary mapping from a kernel to a tarball of pre-generated source and bitstream
declare -A kernel_to_pre_generated_files
kernel_to_pre_generated_files=(
    ["sgemm_large_a10"]="reconfigurable_matmul/pre_generated/ssssmatmul_large_a10_oneapi2023.2_bsp1.2.1.tar.gz"
    ["sgemm_tiny_a10"]="reconfigurable_matmul/pre_generated/ssssmatmul_tiny_a10_oneapi2023.2_bsp1.2.1.tar.gz"
)

if test "${kernel_to_pre_generated_files[$1]+exists}"; then
    tarball=${kernel_to_pre_generated_files[$1]}
    destination=`echo $tarball|cut -f 1 -d "/"`
    echo Expanding $tarball to directory $destination/oneapi, bin and reports ...
    tar xzvf $tarball --directory=$destination --touch
else
    echo "Sorry. No pre-generated files for $1"
fi

cd -
