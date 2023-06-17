from typing import Optional, Union, Sequence, Literal, Tuple
from dataclasses import dataclass
from itertools import product
from copy import deepcopy
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .rbfnn_model import RadialBasisFunctionNeuralNetworkModel



@dataclass
class RBFBatchTrainer:
    input_range: Union[np.ndarray, Sequence[float]]
    inputs: np.ndarray
    reference_outputs: np.ndarray
    epochs: int = 100
    method: Literal['trainlsqr', 'trainlm'] = 'trainlsqr'
    goal: float = 1e-6
    min_grad: float = 1e-10
    train_log_freq: int = 1
    validation_inputs: Optional[np.ndarray] = None
    validation_outputs: Optional[np.ndarray] = None
    mu: float = 10.
    alpha: float = 0.995
    grid_sizes: Sequence[int] = (5,)
    width_ranges: Sequence[Tuple[float, float]] = ((0.1, 10),)
    amplitude_range: Tuple[float, float] = (0.1, 10)
    n_repeat: int = 1
    verbosity: Literal[0, 1, 2] = 2
    path: Path | None = None

    def load_results(self):
        """"""
        results = self.load_results_raw()
        results = pd.DataFrame(results)
        if "centers_init" in results:
            return results
        else:
            return results.groupby(["grid_size", "width_range_min"]).mean().to_xarray()

    def load_results_raw(self):
        """"""
        if self.path is None:
            return []
        elif not self.path.exists():
            return []
        else:
            with self.path.open("rb") as f:
                results = pickle.load(f)
                return results
            
    def save_results(self, results):
        """"""
        if self.path is None:
            return
        with self.path.open("wb") as f:
            pickle.dump(results, f)

    def generate_cases(self):
        """"""        
        for seed, (grid_size, width_range) in enumerate(product(self.grid_sizes, self.width_ranges)):
            for i in range(self.n_repeat):
                yield grid_size, width_range, seed

    def train_batch(self):
        """""" 
        results = self.load_results_raw()
        start_idx = len(results)
        cases = list(self.generate_cases())
        n_total = len(cases)
        cases = cases[start_idx:]
        with tqdm(initial=start_idx, total=n_total, disable=self.verbosity==0) as pbar:
            for result in map(self.train_single, cases):
                pbar.update()
                results.append(result)
                self.save_results(results)
        
        return pd.DataFrame(results).groupby(["grid_size", "width_range_min"]).mean().to_xarray()


    def train_single(self, args: Tuple[int, Tuple[float, float], int | None]):
        """"""
        grid_size, width_range, seed = args
        if seed is not None:
            np.random.seed(seed)
        
        model = RadialBasisFunctionNeuralNetworkModel.new_grid_placement(
            n_inputs=2,
            grid_size=[grid_size, grid_size],
            input_range=self.input_range,
            width_range=width_range,
            amplitude_range=self.amplitude_range,
        )
        model.train(
            inputs=self.inputs,
            reference_outputs=self.reference_outputs,
            validation_inputs=self.validation_inputs,
            validation_outputs=self.validation_outputs,
            epochs=self.epochs,
            goal=self.goal,
            train_log_freq=self.train_log_freq,
            method=self.method,
            mu=self.mu,
            alpha=self.alpha,
            verbose=self.verbosity == 2
        )

        return get_results(model, grid_size, width_range)
    
    def run_init_sensitivity(self, n_samples):
        """"""
        results = self.load_results_raw()
        start_idx = len(results)
        with tqdm(initial=start_idx, total=n_samples, disable=self.verbosity==0) as pbar:
            for i in range(start_idx, n_samples):
                results_lm, results_ols = self.train_init_sensitivity_single(i, True)
                results.append(results_lm)
                results.append(results_ols)
                
                results_lm, results_ols = self.train_init_sensitivity_single(i, False)
                results.append(results_lm)
                results.append(results_ols)

                pbar.update()
                self.save_results(results)
        
        return pd.DataFrame(results)

    def train_init_sensitivity_single(self, seed: int, init_grid: bool):
        """
        Trains the given model using both LB and OLS.
        
        :param seed: Seed used to generate the model.
        :param init_grid: True to initialize RBF centers in uniform grid. Otherwise
                        the centers will be spread around the input domain in
                        a centroidal voronoi tessellation using Lloyds Algorithm.
        :returns: Tuple containing the LM and OLS trained models.
        """

        np.random.seed(seed)
        grid_size = self.grid_sizes[0]

        if init_grid:
            model_lm = RadialBasisFunctionNeuralNetworkModel.new_grid_placement(
                n_inputs=2,
                grid_size=[grid_size, grid_size],
                input_range=self.input_range,
                width_range=self.width_ranges[0],
                amplitude_range=self.amplitude_range,
            )
        else:
            model_lm = RadialBasisFunctionNeuralNetworkModel.new_centroidal_voronoi_tessellation_placement(
                n_hidden=grid_size**2,
                input_range=self.input_range,
                width_range=self.width_ranges[0],
                amplitude_range=self.amplitude_range,
            )
        
        # Copy the model so we know they both have the same initial condition.
        model_ols = deepcopy(model_lm)
        
        model_lm.train(
            inputs=self.inputs,
            reference_outputs=self.reference_outputs,
            validation_inputs=self.validation_inputs,
            validation_outputs=self.validation_outputs,
            epochs=10,
            goal=1e-6,
            train_log_freq=1,
            method="trainlm",
            mu=10.,
            alpha=0.995,
            verbose=False
        )
        
        model_ols.train(
            inputs=self.inputs,
            reference_outputs=self.reference_outputs,
            validation_inputs=self.validation_inputs,
            validation_outputs=self.validation_outputs,
            method="trainlsqr",
            verbose=False
        )
        
        results_lm = get_results(model_lm, grid_size, self.width_ranges[0])
        results_lm['method'] = "lm"
        results_lm['centers_init'] = 'grid' if init_grid else 'loyds'

        results_ols = get_results(model_ols, grid_size, self.width_ranges[0])
        results_ols['method'] = "ols"
        results_ols['centers_init'] = 'grid' if init_grid else 'loyds'

        return results_lm, results_ols


def get_results(model: RadialBasisFunctionNeuralNetworkModel, grid_size: int, width_range: Tuple[float, float]):
    """"""
    best_epoch = model.training_log.min_residual_epoch.item()
    return {
        "grid_size": grid_size,
        "width_range_min": width_range[0],
        "width_range_max": width_range[1],

        "best_epoch": best_epoch,

        "residual_training_mean_absolute": abs(model.training_log.error_training_data.sel(epoch=best_epoch)).mean("i").item(),
        "residual_training_mean": model.training_log.error_training_data.sel(epoch=best_epoch).mean("i").item(),
        "residual_training_jb": abs(model.training_log.error_training_jb.sel(epoch=best_epoch)).mean().item(),

        "residual_validation_mean_absolute": abs(model.training_log.error_validation_data.sel(epoch=best_epoch)).mean("j").item(),
        "residual_validation_mean": model.training_log.error_validation_data.sel(epoch=best_epoch).mean("j").item(),
        "residual_validation_jb": model.training_log.error_validation_jb.sel(epoch=best_epoch).item()
    }
