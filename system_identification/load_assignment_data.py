from pathlib import Path

import xarray as xr
import numpy as np

from scipy.io import loadmat


def load_assignment_data(data_dir_path: Path,
                         f16_data_file_name: str = "F16traindata_CMabV_2017.mat"):
    raw_data = loadmat(str(data_dir_path / f16_data_file_name))

    dt = 0.01
    time = np.arange(raw_data["U_k"].shape[0]) * dt

    data = xr.Dataset(coords={"t": time}, data_vars={
        "alpha_m": (("t", ), raw_data["Z_k"][:, 0]),
        "beta_m": (("t",), raw_data["Z_k"][:, 1]),
        "V_m": (("t",), raw_data["Z_k"][:, 2]),
        "udot": (("t",), raw_data["U_k"][:, 0]),
        "vdot": (("t",), raw_data["U_k"][:, 1]),
        "wdot": (("t",), raw_data["U_k"][:, 2]),
        "c_m": (("t",), raw_data["Cm"].flatten())
    })

    return data


def load_net_example_ff(data_dir_path: Path):
    raw_data = loadmat(str(data_dir_path / "NetExampleFF.mat"), chars_as_strings=True, struct_as_record=True)
    return {
        "input_weights": raw_data['netFF']['IW'][0][0],
        "output_weights": raw_data['netFF']['LW'][0][0],
        "bias_weights": [
            raw_data['netFF']['b'][0][0][0][0],
            raw_data['netFF']['b'][0][0][1][0],
        ],
        "range": raw_data['netFF']['range'][0][0],
    }


