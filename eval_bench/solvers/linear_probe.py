from benchopt import BaseSolver

from nidl.estimators import ssl
from nidl.estimators.probes.model_probes import ModelProbe
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'linear probe'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'
    # and are set to one value of the list.
    parameters = {
        'ssl_class': ['SimCLR'],
        'checkpoint': [None],
        'encoder': [None],
    }
    requirements = []

    def set_objective(self, dataset):
        self.dataset = dataset

        if self.encoder is None:
            from nidl.volume.backbones import resnet18
            self.encoder = resnet18()

        estimator = getattr(ssl, self.ssl_class)(encoder=self.encoder)
        if self.checkpoint is not None:
            estimator.load_checkpoint(self.checkpoint)

        self.model = ModelProbe(
            estimator, probe=Ridge()
        )

    def run(self, _):
        dataloader = DataLoader(self.dataset, batch_size=16)
        self.model.fit(dataloader)

    def get_result(self):
        return dict(model=self.model)
