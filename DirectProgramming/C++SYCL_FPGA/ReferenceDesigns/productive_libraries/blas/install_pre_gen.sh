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

source pre_generated/info.sh

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
    if [[ -d "${demo_dir}" ]]; then
        echo "Directory ${demo_dir} already exists"
    elif [[ -f "${demo_dir}" ]]; then
        echo "${demo_dir} already exists but is not a directory"
        exit 1
    else
        mkdir -p "${demo_dir}"
    fi
    cp pre_generated/${demo} ${demo_dir}
    touch ${demo_dir}/${demo}
else
    echo "Sorry. No pre-generated demo for $1"
    exit 1
fi

cd -
