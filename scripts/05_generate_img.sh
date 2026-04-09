#!/bin/bash

pixi run python ../data_generation/gen_synth_mri.py \
    --labels-dir ../segmentations \
    --out-dir ../synth_mri/ \
    --generation-labels ../segmentations/generation_labels.npy \
    --generation-classes ../segmentations/generation_classes.npy \
    --output-labels ../segmentations/output_labels.npy \
    --prior-means ../segmentations/prior_means.npy \
    --prior-stds ../segmentations/prior_stds.npy \
    --n-per-label 5 