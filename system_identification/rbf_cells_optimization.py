from typing import Optional

import numpy as np

from .rbfnn_model import RadialBasisFunctionNeuralNetworkModel


class RBFCellsOptimizer:
    def __init__(self,
                 inputs: np.ndarray,
                 reference_outputs: np.ndarray,
                 epochs: int = 1,
                 method: str = "trainlm",
                 goal: float = 0.,
                 min_grad: float = 1e-10,
                 train_log_freq: int = 1,
                 validation_inputs: Optional[np.ndarray] = None,
                 validation_outputs: Optional[np.ndarray] = None):
        self.inputs = inputs
        self.reference_outputs = reference_outputs
        self.epochs = epochs
        self.method = method
        self.goal = goal
        self.min_grad = min_grad
        self.train_log_freq = train_log_freq
        self.validation_inputs = validation_inputs
        self.validation_outputs = validation_outputs

    def optimize(self,
                 min_grid_size,
                 max_grid_size):
        pass
        
