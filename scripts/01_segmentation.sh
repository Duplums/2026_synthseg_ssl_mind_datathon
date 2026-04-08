#!/bin/bash

pixi run python ../legacy/SynthSeg/scripts/commands/SynthSeg_predict.py \
    --i ./data/sub-109861838866_T1w.nii.gz \
    --o ./segmentations/ \
    --cpu --threads 55 --parc --robust \
    --vol \
    ./segmentations/volumes.csv \
    --qc ./segmentations/qc.csv