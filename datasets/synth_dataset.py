from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class SynthDataset(Dataset):
    """
    PyTorch Dataset for synthetic MRI data generated from SynthSeg.

    This dataset loads MRI volumes from disk and returns them as PyTorch tensors.

    Parameters
    ----------
    images_dir : str or Path
        Path to directory containing synthetic MRI images (.nii or .nii.gz).

    transform : callable, optional
        Optional transform applied to MRI volumes.
        Signature: transform(image) -> image

    dtype : torch.dtype, default=torch.float32
        Data type for the image tensor.

    Notes
    -----
    - Images are returned with shape (1, D, H, W) (channel-first).
    - No resampling or alignment is performed; assumes preprocessing already done.
    """

    def __init__(
        self,
        images_dir,
        transform=None,
        dtype=torch.float32,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.dtype = dtype

        # Collect image paths
        self.image_paths = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix in [".nii", ".gz"]
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_nifti(self, path):
        """
        Load a NIfTI file and return a numpy array.

        Parameters
        ----------
        path : Path
            Path to NIfTI file.

        Returns
        -------
        array : np.ndarray
            Loaded volume as float32 numpy array.
        """
        return np.asarray(nib.load(str(path)).dataobj)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        image : torch.Tensor
            Tensor of shape (1, D, H, W), dtype=float32.
        """
        img = self._load_nifti(self.image_paths[idx]).astype(np.float32)

        # add channel dimension
        img = np.expand_dims(img, axis=0)

        if self.transform is not None:
            img = self.transform(img)

        img = torch.from_numpy(img).to(self.dtype)

        return img