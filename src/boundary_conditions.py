import numpy as np

from src.lattice_boltzman_equation import vel_to_opp_vel_mapping, get_velocity_sets, get_w_i

from typing import Callable


def get_wall_indices(boundary: np.ndarray):
    index_list = []
    if np.all(boundary[0, :]):
        index_list += [1, 5, 8]
    if np.all(boundary[-1, :]):
        index_list += [3, 6, 7]
    if np.all(boundary[:, 0]):
        index_list += [4, 7, 8]
    if np.all(boundary[:, -1]):
        index_list += [2, 5, 6]
    return np.array(index_list)


def rigid_wall(boundary: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns anonymous function implementing bounce-back boundary condition
    Args:
        boundary: boolean boundary map

    Returns: anonymous function implementing bounce-back boundary condition

    """
    assert boundary.dtype == 'bool'
    change_directions = get_wall_indices(boundary)
    mapping = vel_to_opp_vel_mapping()

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray) -> np.ndarray:
        assert boundary.shape == f_pre_streaming.shape[0:2]
        assert boundary.shape == f_post_streaming.shape[0:2]

        for change_dir in change_directions:
            f_post_streaming[boundary, mapping[change_dir]] = f_pre_streaming[boundary, change_dir]

        return f_post_streaming

    return bc


def moving_wall(boundary: np.ndarray, u_w: np.ndarray, density: np.ndarray) \
        -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns anonymous function implementing moving wall
    Args:
        boundary: boolean boundary map
        u_w: velocity of the wall

    Returns: Returns anonymous function implementing moving wall

    """
    assert boundary.dtype == 'bool'
    c_s = 1 / np.sqrt(3)
    mapping = vel_to_opp_vel_mapping()
    c_i = get_velocity_sets()
    w_i = get_w_i()
    avg_density = np.mean(density)
    change_directions = get_wall_indices(boundary)

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray) -> np.ndarray:
        assert boundary.shape == f_pre_streaming.shape[0:2]
        assert boundary.shape == f_post_streaming.shape[0:2]

        for change_dir in change_directions:
            f_post_streaming[boundary, mapping[change_dir]] = f_pre_streaming[boundary, change_dir] - 2 * w_i[
                change_dir] * avg_density * np.divide(
                c_i[change_dir] @ u_w, c_s ** 2)

        return f_post_streaming

    return bc


def inlet():
    raise NotImplementedError


def outlet():
    raise NotImplementedError


def periodic_with_pressure_variations(boundary: np.ndarray, p_in: float, p_out: float):
    assert boundary.dtype == 'bool'
    assert np.all(boundary[0, :] == boundary[-1, :]) or np.all(boundary[:, 0] == boundary[:, -1])
    from src.lattice_boltzman_equation import equilibrium_distr_func
    c_s = 1 / np.sqrt(3)
    if np.all(boundary[0, :] == boundary[-1, :]):
        density_in = np.divide(p_out, c_s ** 2)
        density_in = np.ones_like(boundary[0, :]) * density_in
        density_out = np.divide(p_out + p_out - p_in, c_s ** 2)
        density_out = np.ones_like(boundary[-1, :]) * density_out
    elif np.all(boundary[:, 0] == boundary[:, -1]):
        density_in = np.divide(p_out, c_s ** 2)
        density_in = np.ones_like(boundary[:, 0]) * density_in
        density_out = np.divide(p_out + p_out - p_in, c_s ** 2)
        density_out = np.ones_like(boundary[:, -1]) * density_out

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray,
           density: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        assert boundary.shape == f_pre_streaming.shape[0:2]
        assert boundary.shape == f_post_streaming.shape[0:2]

        f_eq = equilibrium_distr_func(density_in, velocity[-2, :])
        f_post_streaming[0, ...] = f_eq + (f_post_streaming[-2, ...] - f_eq[-2, ...])
        f_eq = equilibrium_distr_func(density_out, velocity[1, :])
        f_post_streaming[-1, ...] = f_eq + (f_post_streaming[1, ...] - f_eq[1, ...])
        return f_post_streaming

    return bc
