#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BATCH AUGMENTATION OF MRI SEGMENTATION MASKS
================================================================================

DESCRIPTION:
    This script performs data augmentation on MRI segmentation masks to increase
    dataset diversity for synthetic image generation. It applies a combination of
    geometric transformations while preserving anatomical plausibility.

    Transformations applied:
    1. Random Affine: scaling (0.85-1.15) and rotation (up to 10 degrees)
    2. SVF Elastic Deformation: Smooth, diffeomorphic deformations using
       Stationary Velocity Fields for realistic anatomical variability
    3. Random Left-Right Flip: 50% probability horizontal flip

    The script uses GPU acceleration when available and includes automatic
    skip functionality for already-processed files.

USAGE:
    python augment_seg_mask.py \
        --base_dir /path/to/segmentations \
        --pattern "*_synthseg.nii.gz" \
        --num_versions 5 \
        --start_index 1

COMMAND LINE ARGUMENTS:
    --base_dir      : (Required) Base directory for recursive segmentation search
    --pattern       : (Optional) Glob pattern to find segmentation files
                      Default: "*_synthseg.nii.gz"
    --exclude       : (Optional) Patterns to exclude from search
                      Default: ['T2w']
    --num_versions  : (Optional) Number of augmented versions to generate per file
                      Default: 5
    --start_index   : (Optional) Starting index for output file naming
                      Default: 1 (produces _aug1, _aug2, etc.)

CONFIGURABLE PARAMETERS (in code):
    Transform parameters (lines 95-98):
    - scales: Affine scaling range, default (0.85, 1.15)
    - degrees: Maximum rotation angle, default 10
    - nonlin_std: Elastic deformation intensity, default 1.8
    - nonlin_scale: Elastic deformation smoothness, default 0.0625
    - flip_probability: L-R flip probability, default 0.5

OUTPUT:
    Augmented files saved in same directory as source with naming:
    <original_stem>_aug<index>.nii.gz

NOTES:
    - Uses nearest-neighbor interpolation to preserve label integrity
    - GPU memory is cleared every 10 files to prevent memory leaks
    - Corrupted or invalid files are automatically skipped with error logging

REQUIREMENTS:
    - torch
    - torchio
    - numpy

================================================================================
"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser
import torch
import torchio as tio
import glob
import torch.nn.functional as F

# =============================================================================
# GPU SETUP
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")
if torch.cuda.is_available():
    print(f"       GPU: {torch.cuda.get_device_name(0)}")
    print(f"       Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# =============================================================================
# SVF ELASTIC DEFORMATION FUNCTIONS
# =============================================================================

def scale_and_square(velocity, num_steps=7):
    """
    Integrate velocity field using scaling and squaring method.
    Produces a diffeomorphic deformation field from a stationary velocity field.
    
    Args:
        velocity: Stationary velocity field tensor
        num_steps: Number of integration steps (higher = more accurate)
    
    Returns:
        Integrated displacement field
    """
    flow = velocity / (2.0 ** num_steps)
    for _ in range(num_steps):
        flow = flow + warp_field(flow, flow)
    return flow


def warp_field(field, displacement):
    """
    Warp a vector field by a displacement field using bilinear interpolation.
    
    Args:
        field: Vector field to warp
        displacement: Displacement field for warping
    
    Returns:
        Warped vector field
    """
    B, H, W, D, _ = field.shape
    vectors = [torch.arange(0, s, dtype=torch.float32, device=field.device) for s in [H, W, D]]
    grids = torch.meshgrid(vectors, indexing='ij')
    grid = torch.stack(grids, dim=3).unsqueeze(0)
    new_grid = grid + displacement
    
    # Normalize grid to [-1, 1] for grid_sample
    for i, dim_size in enumerate([H, W, D]):
        new_grid[..., i] = 2.0 * (new_grid[..., i] / (dim_size - 1)) - 1.0
    new_grid = new_grid[..., [2, 1, 0]]
    
    field_perm = field.permute(0, 4, 1, 2, 3)
    warped = F.grid_sample(field_perm, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.permute(0, 2, 3, 4, 1)


def generate_svf_deformation(shape, nonlin_std=3.0, nonlin_scale=0.0625, device=device):
    """
    Generate a smooth diffeomorphic deformation using Stationary Velocity Fields.
    
    The deformation is generated at low resolution and upsampled to ensure
    smooth, anatomically plausible transformations.
    
    Args:
        shape: Target volume shape [H, W, D]
        nonlin_std: Standard deviation of deformation intensity
        nonlin_scale: Scale factor for low-resolution field (smaller = smoother)
        device: Computation device (CPU/CUDA)
    
    Returns:
        Deformation flow field tensor
    """
    # Generate low-resolution velocity field
    small_shape = [max(int(s * nonlin_scale), 4) for s in shape]
    std = torch.rand(1, device=device).item() * nonlin_std
    small_field = torch.randn([1] + small_shape + [3], device=device) * std
    
    # Upsample through intermediate resolution for smoother result
    mid_shape = [max(s // 2, small_shape[i]) for i, s in enumerate(shape)]
    mid_field = F.interpolate(
        small_field.permute(0, 4, 1, 2, 3),
        size=mid_shape,
        mode='trilinear',
        align_corners=True
    ).permute(0, 2, 3, 4, 1)
    
    # Integrate velocity field to get displacement
    flow = scale_and_square(mid_field, num_steps=7)
    
    # Upsample to full resolution
    final_flow = F.interpolate(
        flow.permute(0, 4, 1, 2, 3),
        size=shape,
        mode='trilinear',
        align_corners=True
    ).permute(0, 2, 3, 4, 1)
    
    return final_flow[0]


def apply_deformation(image_data, flow):
    """
    Apply deformation field to image data using nearest-neighbor interpolation.
    Nearest-neighbor is used to preserve discrete label values in segmentation masks.
    
    Args:
        image_data: Image tensor [C, H, W, D]
        flow: Deformation flow field [H, W, D, 3]
    
    Returns:
        Deformed image tensor
    """
    C, H, W, D = image_data.shape
    image_batch = image_data.unsqueeze(0)
    flow_batch = flow.unsqueeze(0)
    
    # Create sampling grid
    vectors = [torch.arange(0, s, dtype=torch.float32, device=image_data.device) for s in [H, W, D]]
    grids = torch.meshgrid(vectors, indexing='ij')
    grid = torch.stack(grids, dim=3).unsqueeze(0)
    
    # Apply displacement
    new_grid = grid + flow_batch
    
    # Normalize to [-1, 1]
    for i, dim_size in enumerate([H, W, D]):
        new_grid[..., i] = 2.0 * (new_grid[..., i] / (dim_size - 1)) - 1.0
    new_grid = new_grid[..., [2, 1, 0]]
    
    # Sample with nearest-neighbor to preserve label integrity
    warped = F.grid_sample(
        image_batch,
        new_grid,
        mode='nearest',
        padding_mode='border',
        align_corners=True
    )
    return warped[0]


class SVFElasticDeformation:
    """
    TorchIO-compatible transform for SVF-based elastic deformation.
    Wraps the SVF deformation functions for use in TorchIO pipelines.
    """
    
    def __init__(self, nonlin_std=3.0, nonlin_scale=0.0625, device=device):
        """
        Args:
            nonlin_std: Maximum deformation intensity
            nonlin_scale: Smoothness scale (smaller = smoother deformations)
            device: Computation device
        """
        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale
        self.device = device
    
    def __call__(self, image):
        """Apply deformation to a TorchIO image."""
        data = image.data.to(self.device)
        shape = list(data.shape[1:])
        flow = generate_svf_deformation(shape, self.nonlin_std, self.nonlin_scale, device=self.device)
        warped_data = apply_deformation(data, flow)
        return tio.LabelMap(tensor=warped_data.cpu(), affine=image.affine)


# =============================================================================
# TRANSFORMATION PIPELINE
# =============================================================================
# Parameters tuned for realistic augmentation without overfitting:
# - Moderate scaling and rotation to simulate scanner positioning variability
# - Soft elastic deformation to simulate natural anatomical variability
# - Random L-R flip for hemisphere symmetry augmentation

transform = tio.Compose([
    tio.RandomAffine(
        scales=(0.85, 1.15),      # Scaling range
        degrees=10,               # Max rotation degrees
        translation=0,            # No translation
        image_interpolation='nearest'  # Preserve labels
    ),
    SVFElasticDeformation(
        nonlin_std=1.8,           # Deformation intensity
        nonlin_scale=0.0625,      # Smoothness factor
        device=device
    ),
    tio.RandomFlip(
        axes=('LR',),             # Left-Right axis only
        flip_probability=0.5
    ),
])


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def augment_batch(base_dir, pattern='*_seg.nii.gz', exclude_patterns=None, num_versions=5, start_index=1):
    """
    Augment all segmentation masks found recursively in base_dir.
    
    Args:
        base_dir: Base directory for recursive file search
        pattern: Glob pattern for segmentation files
        exclude_patterns: List of patterns to exclude from processing
        num_versions: Number of augmented versions per original file
        start_index: Starting index for output file naming
    """
    
    if exclude_patterns is None:
        exclude_patterns = ['T2w']
    
    print(f"\n{'='*80}")
    print("FILE SEARCH")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Search pattern: {pattern}")
    print(f"Excluded patterns: {', '.join(exclude_patterns)}")
    
    # Recursive file search
    search_pattern = os.path.join(base_dir, '**', pattern)
    all_files = glob.glob(search_pattern, recursive=True)
    
    # Apply exclusion filters
    seg_files = []
    for f in all_files:
        basename = os.path.basename(f)
        if not any(excl in basename for excl in exclude_patterns):
            seg_files.append(f)
    
    seg_files = sorted(seg_files)
    
    print(f"\nFound {len(seg_files)} segmentation files")
    
    if len(seg_files) == 0:
        print("\nERROR: No files found!")
        return
    
    print(f"\n{'='*80}")
    print(f"AUGMENTATION - Generating aug_{start_index} to aug_{start_index + num_versions - 1}")
    print(f"{'='*80}\n")
    
    total_processed = 0
    total_skipped = 0
    failed_files = []
    
    for idx, seg_file in enumerate(seg_files, 1):
        seg_file = Path(seg_file)
        
        # Display relative path for cleaner output
        try:
            rel_path = seg_file.relative_to(base_dir)
        except ValueError:
            rel_path = seg_file
        
        print(f"  [{idx}/{len(seg_files)}] {rel_path}")
        output_dir = seg_file.parent
        
        # Check if all versions already exist (skip if complete)
        stem = seg_file.stem.replace(".nii", "")
        last_version_num = start_index + num_versions - 1
        last_aug = output_dir / f"{stem}_aug{last_version_num}.nii.gz"
        
        if last_aug.exists():
            print(f"    [SKIP] Versions up to {last_version_num} already exist")
            total_skipped += 1
            continue
        
        # Process file with error handling
        try:
            image = tio.LabelMap(seg_file)
            
            # Generate all augmented versions
            for i in range(start_index, start_index + num_versions):
                augmented = transform(image)
                new_filename = f"{stem}_aug{i}.nii.gz"
                out_path = output_dir / new_filename
                augmented.save(out_path)
                print(f"    [OK] {new_filename}")
            
            total_processed += 1
        
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:150]
            
            print(f"    [ERROR] {error_type}")
            print(f"    [SKIP] File corrupted or invalid")
            
            failed_files.append({
                'file': str(seg_file),
                'error_type': error_type,
                'error_msg': error_msg
            })
            continue
        
        # Clear GPU cache periodically to prevent memory leaks
        if idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {total_processed}")
    print(f"Skipped (already exist): {total_skipped}")
    print(f"Failed (errors): {len(failed_files)}")
    print(f"Versions per file: {num_versions}")
    print(f"Total files generated: {total_processed * num_versions}")
    
    # Print error details if any
    if failed_files:
        print(f"\n{'='*80}")
        print("FAILED FILES")
        print(f"{'='*80}\n")
        
        for idx, failed in enumerate(failed_files, 1):
            print(f"{idx}. {failed['file']}")
            print(f"   Error type: {failed['error_type']}")
            print(f"   Message: {failed['error_msg']}")
            print()


def main():
    """Main entry point with argument parsing."""
    parser = ArgumentParser(
        description="Batch augmentation of MRI segmentation masks (GPU-Accelerated)",
        epilog="Works with any dataset directory structure"
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory for recursive segmentation file search"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_synthseg.nii.gz",
        help="Glob pattern for segmentation files (default: *_synthseg.nii.gz)"
    )
    
    parser.add_argument(
        "--exclude",
        nargs='+',
        default=['T2w'],
        help="Patterns to exclude from filenames (default: T2w)"
    )
    
    parser.add_argument(
        "--num_versions",
        type=int,
        default=5,
        help="Number of augmented versions per segmentation (default: 5)"
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Starting index for output file naming (default: 1)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"ERROR: Directory not found: {args.base_dir}")
        sys.exit(1)
    
    augment_batch(args.base_dir, args.pattern, args.exclude, args.num_versions, args.start_index)


if __name__ == "__main__":
    main()
