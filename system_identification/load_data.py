from pathlib import Path
import xarray as xr
import numpy as np

from scipy.io import loadmat


def load_data(data_dir_path: Path,
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


if __name__ == '__main__':
    load_data(Path("/home/adewindt/repos/system_identification/assignment"))

