import numpy as np

from src.lattice_boltzman_equation import vel_to_opp_vel_mapping, get_velocity_sets, get_w_i

from typing import Callable
from copy import deepcopy


def rigid_wall(boundary: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    assert boundary.dtype == 'bool'

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray, *argv) -> np.ndarray:
        assert boundary.shape == f_pre_streaming.shape[0:2]
        assert boundary.shape == f_post_streaming.shape[0:2]

        mapping = vel_to_opp_vel_mapping()
        for i in range(f_pre_streaming.shape[-1]):
            f_post_streaming[boundary, mapping[i]] = f_pre_streaming[boundary, i]
        return f_post_streaming

    return bc


def moving_wall(boundary: np.ndarray, u_w: np.ndarray) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    assert boundary.dtype == 'bool'
    c_s = 1 / np.sqrt(3)

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray, density: np.ndarray, *argv) -> np.ndarray:
        assert boundary.shape == f_pre_streaming.shape[0:2]
        assert boundary.shape == f_post_streaming.shape[0:2]

        mapping = vel_to_opp_vel_mapping()
        c_i = get_velocity_sets()
        w_i = get_w_i()

        for i in range(f_pre_streaming.shape[-1]):
            f_post_streaming[boundary, mapping[i]] = f_pre_streaming[boundary, i] - 2 * w_i[i] * density[
                boundary] * np.divide(c_i[i] @ u_w, c_s ** 2)
        return f_post_streaming

    return bc


def inlet():
    raise NotImplementedError


def outlet():
    raise NotImplementedError


def periodic_with_pressure_variations():
    raise NotImplementedError
