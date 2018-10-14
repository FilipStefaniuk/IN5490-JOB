#!/bin/bash

OUTPUT_DIR=$1

shift

for CONFIG in $@; do
    python3 train_cartpole.py --tensorboard $OUTPUT_DIR $CONFIG
done
