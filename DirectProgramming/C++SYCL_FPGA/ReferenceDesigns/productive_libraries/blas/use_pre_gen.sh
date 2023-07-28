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
        tar xzvf $entry --directory=$directory
    done
done

