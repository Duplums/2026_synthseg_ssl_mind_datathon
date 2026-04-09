from benchopt import BaseDataset
from benchopt.config import get_data_path

from nidl.datasets.openbhb import OpenBHB
from torch.utils.data import Subset


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "OpenBHB"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {'debug': [False]}

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def prepare(self):
        # Download the data once for all
        root = get_data_path("openbhb")
        for split in ['train', 'val']:
            OpenBHB(
                root=root,
                streaming=False,
                modality='quasiraw',
                split=split
            )

    def get_data(self):

        root = get_data_path("openbhb")
        dataset = OpenBHB(
            root=root,
            streaming=True,
            modality='quasiraw',
            split='train'
        )
        dataset_val = OpenBHB(
            root=root,
            streaming=True,
            modality='quasiraw',
            split='val'
        )

        if self.debug:
            dataset = Subset(dataset, range(3))
            dataset_val = Subset(dataset_val, range(2))

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(dataset=dataset, dataset_val=dataset_val)
