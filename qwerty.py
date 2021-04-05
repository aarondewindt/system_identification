from itertools import product
from pathlib import Path
from tqdm.auto import trange

import numpy as np
import xarray as xr
import xarray.ufuncs as xf
import matplotlib.pyplot as plt

from system_identification.ffnn import FeedForwardNeuralNetwork, TrainingParameters
from system_identification.load_assignment_data import load_net_example_ff

nn = FeedForwardNeuralNetwork.new(
    n_inputs=2,
    n_outputs=1,
    n_hidden=2,
    range=[[-1, 1], [-1, 1]],
    training_parameters=TrainingParameters(
        epochs=1000,
        goal=0,
        min_grad=1e-10,
        mu=0.001,
    ),
)


def f(x):
    return -0.7 * np.tanh(x[0]) + np.tanh(10 * x[1]) + 1


inputs = np.random.uniform(-5, 5, (10000, 2))
reference_outputs = np.array(list(map(f, inputs))).reshape(10000, 1)

nn.back_propagation(inputs, reference_outputs, epochs=10)
