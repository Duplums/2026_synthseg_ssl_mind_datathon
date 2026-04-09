from benchopt import BaseObjective

from torch.utils.data import DataLoader

from nidl.metrics.regression import pearson_r, r2_score, median_absolute_error


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    name = "Eval SSL SynthSeg"
    url = "https://github.com/Duplums/eval-ssl-synthseg"
    parameters = {
        'random_state': [32],
        'batch_size': [64],
    }

    # List of packages needed to run the benchmark.
    requirements = ['pip::git+https://github.com/neurospin-deepinsight/nidl']

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.9"
    sampling_strategy = "run_once"

    def set_data(self, dataset, dataset_val):
        self.dataset = dataset
        self.dataset_val = dataset_val

    def evaluate_result(self, model):
        scoring = {
            'r2_score': r2_score,
            'mae': median_absolute_error,
            'pearson_r': pearson_r
        }

        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size
        )
        score_train = model.score(dataloader, scoring=scoring)
        dataloader_val = DataLoader(
            self.dataset_val, batch_size=self.batch_size
        )
        score_val = model.score(dataloader_val, scoring=scoring)

        # This method can return many metrics in a dictionary.
        return dict(
            **{
                f"{k}": v for k, v in score_train.items()
            },
            **{
                f"{k}_val": v for k, v in score_val.items()
            }
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.

        class DummyTransformer:
            def score(self, dataloader): return dict(score=0.5)
        clf = DummyTransformer()
        return dict(model=clf)

    def get_objective(self):

        return dict(dataset=self.dataset)
