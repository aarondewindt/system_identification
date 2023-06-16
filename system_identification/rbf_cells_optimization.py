from typing import Optional, Union, Sequence

import numpy as np


from .rbfnn_model import RadialBasisFunctionNeuralNetworkModel


class RBFCellsOptimizer:
    def __init__(self,
                 input_range: Union[np.ndarray, Sequence[float]],
                 inputs: np.ndarray,
                 reference_outputs: np.ndarray,                 
                 epochs: int = 100,
                 goal: float = 1e-6,
                 min_grad: float = 1e-10,
                 train_log_freq: int = 1,
                 validation_inputs: Optional[np.ndarray] = None,
                 validation_outputs: Optional[np.ndarray] = None,
                 min_n_hidden: int=1,
                 max_n_hidden: int=500,
                 mu=10.,
                 alpha=0.995):
        self.inputs = inputs
        self.reference_outputs = reference_outputs
        self.epochs = epochs
        self.goal = goal
        self.min_grad = min_grad
        self.train_log_freq = train_log_freq
        self.input_range = input_range
        self.validation_inputs = validation_inputs
        self.validation_outputs = validation_outputs
        self.min_n_hidden = min_n_hidden
        self.max_n_hidden = max_n_hidden
        self.mu = mu
        self.alpha = alpha


    def train_single(self, n_hidden: int, verbose=True):
        model = RadialBasisFunctionNeuralNetworkModel.new_grid_placement(
            n_inputs=2,
            grid_size=[n_hidden, n_hidden],
            input_range=self.input_range,
            rbf_width=8.71,
            rbf_amplitude=1,
        )
        
        # model = RadialBasisFunctionNeuralNetworkModel.new_centroidal_voronoi_tessellation_placement(
        #     n_hidden=n_hidden,
        #     input_range=self.input_range,
        #     width_range=(0.1, 20),
        #     amplitude_range=(1., 1.),
        # )

        # model.train(
        #     inputs=self.inputs,
        #     reference_outputs=self.reference_outputs,
        #     validation_inputs=self.validation_inputs,
        #     validation_outputs=self.validation_outputs,
        #     method="trainlsqr",
        # )

        # model = model.clone_clean()

        model.train(
            inputs=self.inputs,
            reference_outputs=self.reference_outputs,
            validation_inputs=self.validation_inputs,
            validation_outputs=self.validation_outputs,
            epochs=self.epochs,
            goal=self.goal,
            train_log_freq=self.train_log_freq,
            method="trainlm",
            mu=self.mu,
            alpha=self.alpha,
            verbose=verbose
        )

        return model

    def optimize(self):
        pass
