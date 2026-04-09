#!/bin/bash

pixi run python ../data_generation/build_generation_arrays.py \
    --labels-dir ../segmentations \
    --out-dir ../segmentations