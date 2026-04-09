#!/bin/bash

pixi run python ../data_generation/estimate_prior_intensities.py \
        --original_base_dir ../data \
        --segmented_base_dir ../segmentations \
        --labels_list ../segmentations/generation_labels.npy \
        --out_dir ../segmentations