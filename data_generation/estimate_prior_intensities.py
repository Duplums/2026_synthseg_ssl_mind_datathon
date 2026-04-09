#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EXTRACT TISSUE STATISTICS FROM MRI IMAGES
================================================================================

DESCRIPTION:
    This script extracts intensity statistics (mean, standard deviation) for each
    tissue label from real MRI images by matching them with their corresponding
    segmentation masks. The extracted statistics are saved in a npy file that
    will be used by the synthetic image generator to produce realistic MRI images.

    Under the hood, the script relies on `build_intensity_stats` from SynthSeg to 
    compute the statistics.


USAGE:
    python estimate_prior_intensities.py \
        --original_base_dir /path/to/original/mri \
        --segmented_base_dir /path/to/segmentations \
        --out_dir /path/to/output/tissue_statistics 

COMMAND LINE ARGUMENTS:
    --original_base_dir   : (Required) Base directory containing original MRI images
    --segmented_base_dir  : (Required) Base directory containing segmentation masks
                            Must have parallel structure to original_base_dir
    --out_dir          : (Required) Output path for npy statistics files

OUTPUT:
    - npy files with tissue statistics, named:
      * prior_means.npy (length (2, K) where K is the number of tissues/labels and 2 corresponds to mean and std)
      * prior_stds.npy (same length as `prior_means`)

================================================================================
"""

import os
import sys
import glob
from argparse import ArgumentParser
sys.path.append("../legacy/SynthSeg")
from SynthSeg.estimate_priors import build_intensity_stats

def batch_extract(original_base_dir, segmented_base_dir, labels_list, out_dir):
    """
    Batch extraction of tissue statistics from all MRI images found recursively.
    
    Args:
        original_base_dir: Base directory with original MRI images
        segmented_base_dir: Base directory with segmentation masks (parallel structure)
        labels_list: Path to npy file listing all label values to extract statistics from.
        out_dir: Output directory for npy files
    """
    
    print(f"\n{'='*80}")
    print("SEARCHING MRI IMAGES")
    print(f"{'='*80}")
    print(f"Original images directory: {original_base_dir}")
    print(f"Segmentations directory: {segmented_base_dir}")
    
    # Find all MRI images recursively
    pattern = os.path.join(original_base_dir, '**', '*.nii.gz')
    all_files = glob.glob(pattern, recursive=True)
    
    original_files = sorted(all_files)
    
    print(f"\nFound {len(original_files)} MRI images")
    
    if len(original_files) == 0:
        print("\nERROR: No images found!")
        return
    
    print(f"\n{'='*80}")
    print("EXTRACTING TISSUE STATISTICS")
    print(f"{'='*80}\n")
    
    # Extract statistics from all images
    prior_means, prior_stds = build_intensity_stats(
        original_base_dir,
        segmented_base_dir,
        out_dir,
        estimation_labels=labels_list,
    )
    
    if len(prior_means) == 0:
        print("\n[WARNING] No statistics extracted!")
        print("          Verify that segmentations exist and match the original image structure")
        return
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"[OK] Statistics extracted for {prior_means.shape[1]} tissues/labels")
    print(f"Saved to: {out_dir}")
    
def main():
    """Main entry point with argument parsing."""
    parser = ArgumentParser(
        description="Batch tissue statistics extraction from real MRI images",
        epilog="Works with any dataset directory structure"
    )
    
    parser.add_argument(
        "--original_base_dir",
        type=str,
        required=True,
        help="Base directory containing original MRI images"
    )
    
    parser.add_argument(
        "--segmented_base_dir",
        type=str,
        required=True,
        help="Base directory containing segmentation masks (parallel structure to originals)"
    )
    
    parser.add_argument(
        "--labels_list",
        type=str,
        required=True,
        help="Path to npy file listing all label values to extract statistics from"
    )
    
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Base path for output npy statistics files"
    )
    
    
    args = parser.parse_args()
    
    # Verify directories exist
    if not os.path.exists(args.original_base_dir):
        print(f"ERROR: Original images directory not found: {args.original_base_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.segmented_base_dir):
        print(f"ERROR: Segmentations directory not found: {args.segmented_base_dir}")
        sys.exit(1)
    
    # Execute batch extraction
    batch_extract(
        args.original_base_dir, 
        args.segmented_base_dir, 
        args.labels_list,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
