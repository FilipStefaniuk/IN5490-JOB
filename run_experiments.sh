#!/bin/bash

for file in "$2"*; do
    file=${file##*/}
    file=${file%.*}
    
    sbatch $1 "$2" "${3}${file}"
    # $1 "$2" "${3}${file}"
done