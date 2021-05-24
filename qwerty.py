from itertools import product
from pathlib import Path
from tqdm.auto import trange
import pickle

import numpy as np
import xarray as xr
import xarray.ufuncs as xf
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from system_identification.ffnn import FeedForwardNeuralNetwork, TrainingParameters
from system_identification.load_assignment_data import load_net_example_ff

nn = FeedForwardNeuralNetwork.new(
    n_inputs=2,
    n_outputs=1,
    n_hidden=12,
    range=[[-1, 1], [-1, 1]],
    log_dir="./ffnn_exp",
    training_parameters=TrainingParameters(
        epochs=1000,
        goal=0,
        min_grad=1e-10,
        mu=0.001,
    ),
)


def f(x):
    return -0.8 * np.tanh(x[0] * 3) + x[1]**2 + 1


inputs = np.random.uniform(-5, 5, (1000, 2, 1))
reference_outputs = np.array(list(map(f, inputs))).reshape(1000, 1, 1)

nn.train(inputs, reference_outputs, epochs=10_000, train_log_freq=10)
