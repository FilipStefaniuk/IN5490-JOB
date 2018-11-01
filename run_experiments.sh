#!/bin/bash

for file in "$3"*; do
    basename=${file##*/}
    basename=${basename%.*}
    
    sbatch $1 "${2}${basename}" "${file}"
done
