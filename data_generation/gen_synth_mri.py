#!/usr/bin/env python3
"""
Generate synthetic brain MRIs from SynthSeg label maps using precomputed
GMM priors (prior_means.npy / prior_stds.npy).

This script uses SynthSeg.BrainGenerator, which wraps labels_to_image_model.
It generates several images per input label map and saves both:
  - the synthetic image
  - the corresponding output label map returned by the generator

Expected inputs
---------------
- labels_dir: folder with your augmented segmentation masks (.nii/.nii.gz/.mgz)
- generation_labels: 1D npy array listing all label values present in the masks
- generation_classes: 1D npy array, same length as generation_labels,
  mapping each label to an intensity class in [0, K-1]
- prior_means: npy array of shape (2, K) or (2*n_channels, K)
- prior_stds:  npy array of shape (2, K) or (2*n_channels, K)

Notes
-----
- Use prior_distribution='normal' if your prior_means/prior_stds came from
  estimated Gaussian statistics.
"""

from pathlib import Path
import argparse
import numpy as np
import sys
sys.path.append("../legacy/SynthSeg")
from SynthSeg.brain_generator import BrainGenerator
from ext.lab2im import utils


def save_volume(array, ref_path, out_path):
    """
    Save a generated image or label map using the affine/header of ref_path.
    """
    _, aff, header = utils.load_volume(str(ref_path), im_only=False)
    utils.save_volume(array, aff, header, str(out_path))


def main(args):
    labels_dir = Path(args.labels_dir)
    out_img_dir = Path(args.out_dir) / "images"
    out_lab_dir = Path(args.out_dir) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    # Build the generator once. BrainGenerator wraps labels_to_image_model and
    # samples new contrasts each time generate_image() is called.
    generator = BrainGenerator(
        labels_dir=str(labels_dir),
        generation_labels=args.generation_labels,
        n_neutral_labels=args.n_neutral_labels,
        output_labels=args.output_labels,
        batchsize=1,
        n_channels=args.n_channels,
        target_res=None,                  # keep native label-map resolution
        output_shape=None,                # no random cropping
        output_div_by_n=None,
        prior_distributions=args.prior_distribution,
        generation_classes=args.generation_classes,
        prior_means=args.prior_means,
        prior_stds=args.prior_stds,
        use_specific_stats_for_channel=args.use_specific_stats_for_channel,
        mix_prior_and_random=False,
        flipping=False,
        scaling_bounds=args.scaling_bounds,
        rotation_bounds=args.rotation_bounds,
        shearing_bounds=args.shearing_bounds,
        translation_bounds=False,
        nonlin_std=args.nonlin_std,
        nonlin_scale=args.nonlin_scale,
        randomise_res=False,              # usually best if you want native-grid outputs
        bias_field_std=args.bias_field_std,
        bias_scale=args.bias_scale,
        return_gradients=False,
    )

    label_paths = utils.list_images_in_folder(str(labels_dir))
    n_label_maps = len(label_paths)

    # BrainGenerator picks a label map internally at each call.
    # To generate several images per label map, we loop n_label_maps * n_per_label.
    counts = {Path(p).stem.replace(".nii", ""): 0 for p in label_paths}

    for _ in range(n_label_maps * args.n_per_label):
        image_batch, label_batch = generator.generate_brain()

        # batchsize=1
        image = np.asarray(image_batch[0]).squeeze()
        label = np.asarray(label_batch[0]).squeeze()

        # Recover which label map was sampled in this iteration.
        # BrainGenerator samples from labels_dir; with batchsize=1 and uniform
        # sampling, we save outputs with a simple running index.
        # If you need strict per-file control, see note below.
        sampled_name = f"sample_{sum(counts.values()):05d}"
        img_out = out_img_dir / f"{sampled_name}_synthetic.nii.gz"
        lab_out = out_lab_dir / f"{sampled_name}_labels.nii.gz"

        # Save with the affine/header of the first label map shape family.
        # BrainGenerator returns outputs in native label-map space.
        ref_path = label_paths[0]
        save_volume(image, ref_path, img_out)
        save_volume(label.astype(np.int16), ref_path, lab_out)

        print(f"Saved {img_out.name} and {lab_out.name}")

    print(f"\nDone. Outputs written to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic MRIs from SynthSeg label maps."
    )
    parser.add_argument("--labels-dir", required=True,
                        help="Folder containing input label maps.")
    parser.add_argument("--out-dir", required=True,
                        help="Output folder.")
    parser.add_argument("--generation-labels", required=True,
                        help="Path to generation_labels.npy")
    parser.add_argument("--generation-classes", required=True,
                        help="Path to generation_classes.npy")
    parser.add_argument("--prior-means", required=True,
                        help="Path to prior_means.npy")
    parser.add_argument("--prior-stds", required=True,
                        help="Path to prior_stds.npy")
    parser.add_argument("--output-labels", default=None,
                        help="Optional path to output_labels.npy")
    parser.add_argument("--n-per-label", type=int, default=5,
                        help="Number of synthetic images to generate per label map.")
    parser.add_argument("--n-channels", type=int, default=1,
                        help="Number of channels to synthesize.")
    parser.add_argument("--prior-distribution", choices=["uniform", "normal"],
                        default="normal",
                        help="Type of prior used to sample GMM parameters.")
    parser.add_argument("--use-specific-stats-for-channel", action="store_true",
                        help="Use channel-specific blocks in prior arrays.")
    parser.add_argument("--n-neutral-labels", type=int, default=None,
                        help="Needed only if flipping=True and labels are left/right ordered.")
    parser.add_argument("--scaling-bounds", type=float, default=0.15,
                        help="Random scaling bound.")
    parser.add_argument("--rotation-bounds", type=float, default=10.0,
                        help="Random rotation bound in degrees.")
    parser.add_argument("--shearing-bounds", type=float, default=0.01,
                        help="Random shearing bound.")
    parser.add_argument("--nonlin-std", type=float, default=3.0,
                        help="Std of nonlinear deformation field.")
    parser.add_argument("--nonlin-scale", type=float, default=0.03,
                        help="Scale of nonlinear deformation field.")
    parser.add_argument("--bias-field-std", type=float, default=0.5,
                        help="Std of multiplicative bias field.")
    parser.add_argument("--bias-scale", type=float, default=0.025,
                        help="Spatial scale of bias field.")

    main(parser.parse_args())