#!/bin/bash

pixi run python ../data_generation/augment_seg_mask.py \
        --base_dir ../segmentations \
        --pattern "*_synthseg.nii.gz" \
        --num_versions 5 \
        --start_index 1