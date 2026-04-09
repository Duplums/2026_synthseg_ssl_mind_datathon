#!/usr/bin/env python3
"""
Build SynthSeg generation arrays from predicted segmentation maps.

This script extracts all unique labels from a directory of label maps
(e.g. SynthSeg outputs) and creates:

- generation_labels.npy
- output_labels.npy
- generation_classes.npy

These arrays are guaranteed to be consistent with the data and avoid
indexing errors during label-to-image generation.

Usage
-----
python build_generation_arrays.py \
    --labels-dir /path/to/synthseg_predictions \
    --out-dir /path/to/output_arrays

Notes
-----
- generation_labels: sorted list of all labels found in the dataset
- output_labels: identical to generation_labels (no remapping)
- generation_classes: one class per label (simplest valid setup)

This setup is safe but not optimal statistically. You can later merge
classes if needed.
"""

from pathlib import Path
import argparse
import numpy as np
import nibabel as nib


def extract_labels(labels_dir):
    """
    Extract unique label values from all NIfTI files in a directory.

    Parameters
    ----------
    labels_dir : Path
        Directory containing label maps.

    Returns
    -------
    labels : np.ndarray
        Sorted array of unique integer labels.
    """
    all_labels = set()

    for path in sorted(labels_dir.glob("*.nii*")):
        data = np.asarray(nib.load(str(path)).dataobj)
        unique = np.unique(data).astype(int)
        all_labels.update(unique.tolist())

        print(f"{path.name}: {len(unique)} labels")

    labels = np.array(sorted(all_labels), dtype=np.int32)
    return labels


def main(args):
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting labels...")
    generation_labels = extract_labels(labels_dir)

    print(f"\nTotal unique labels found: {len(generation_labels)}")
    print("First labels:", generation_labels[:20])
    print("Last labels:", generation_labels[-20:])

    # Output labels: identity mapping
    output_labels = generation_labels.copy()

    # Generation classes: one class per label (safe default)
    generation_classes = np.arange(len(generation_labels), dtype=np.int32)

    # Save
    np.save(out_dir / "generation_labels.npy", generation_labels)
    np.save(out_dir / "output_labels.npy", output_labels)
    np.save(out_dir / "generation_classes.npy", generation_classes)

    print("\nSaved:")
    print(out_dir / "generation_labels.npy")
    print(out_dir / "output_labels.npy")
    print(out_dir / "generation_classes.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build SynthSeg generation arrays from label maps"
    )
    parser.add_argument("--labels-dir", required=True,
                        help="Directory with SynthSeg prediction label maps")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for .npy files")

    main(parser.parse_args())