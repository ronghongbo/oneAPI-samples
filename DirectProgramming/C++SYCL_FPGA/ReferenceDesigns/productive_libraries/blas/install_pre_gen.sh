#!/bin/bash
directories=(
    reconfigurable_matmul
)

array_to_read=("${directories[@]}")
index=0
while [ "$index" -lt "${#array_to_read[*]}" ]; do
    directory=${array_to_read[$index]}
    let index=index+1
    for entry in $directory/pre_generated/*.tar.gz
    do
        bn=$(basename -- "$entry")
        bn_no_extension=${bn%%.*}
        echo Installing $bn_no_extension files to directory $directory ...
        tar xzvf $entry --directory=$directory --touch
        echo 
    done
done

