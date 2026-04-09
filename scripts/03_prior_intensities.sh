#!/bin/bash

pixi run python ../legacy/SynthSeg/scripts/estimate_prior_intensities.py \
        --original_base_dir ../data \
        --segmented_base_dir ../segmentations \
        --out_dir ../segmentations