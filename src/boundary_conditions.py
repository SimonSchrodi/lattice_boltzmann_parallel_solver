import numpy as np

from lattice_boltzman_equation import vel_to_opp_vel_mapping, get_velocity_sets, get_w_i, equilibrium_distr_func

from typing import Callable, Tuple
import operator


def get_wall_indices(boundary: np.ndarray) -> np.ndarray:
    index_list = []
    if np.all(boundary[0, :]):
        index_list += [1, 5, 8]
    elif np.all(boundary[-1, :]):
        index_list += [3, 6, 7]
    elif np.all(boundary[:, 0]):
        index_list += [4, 7, 8]
    elif np.all(boundary[:, -1]):
        index_list += [2, 5, 6]
    return np.array(index_list)


def get_corner_indices(boundary: np.ndarray) -> np.ndarray:
    assert len(boundary.shape) == 2

    indices = np.argwhere(boundary)
    upper_left_corner = (np.amin(indices[:, 0]), np.amax(indices[:, 1]))
    # above_upper_left_corner = tuple(map(operator.add, upper_left_corner, (0, 1)))
    upper_right_corner = (np.amax(indices[:, 0]) + 1, np.amax(indices[:, 1]))
    # above_upper_right_corner = tuple(map(operator.add, upper_right_corner, (0, 1)))
    lower_left_corner = (np.amin(indices[:, 0]), np.amin(indices[:, 1]))
    # below_lower_left_corner = tuple(map(operator.add, lower_left_corner, (0, -1)))
    lower_right_corner = (np.amax(indices[:, 0]) + 1, np.amin(indices[:, 1]))
    # below_lower_right_corner = tuple(map(operator.add, lower_right_corner, (0, -1)))
    return np.array(
        [
            (upper_left_corner, (1, 8)),
            (lower_left_corner, (1, 5)),
            (upper_right_corner, (3, 7)),
            (lower_right_corner, (3, 6))
        ]
    )


def remove_corner_indices_from_boundary(boundary: np.ndarray, corner_indices: np.ndarray) -> np.ndarray:
    assert len(boundary.shape) == 2
    for corner_index, _ in corner_indices:
        boundary[corner_index[0], corner_index[1]] = False
    return boundary


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


def rigid_object(boundary: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    assert boundary.dtype == 'bool'

    corner_indices = get_corner_indices(boundary)
    boundary_wo_corner = remove_corner_indices_from_boundary(boundary, corner_indices)
    boundary_wo_corner_left = boundary_wo_corner
    boundary_wo_corner_right = np.roll(boundary_wo_corner, 1, axis=0)
    change_directions_left = [1, 5, 8]
    change_directions_right = [3, 6, 7]
    mapping = vel_to_opp_vel_mapping()

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray) -> np.ndarray:
        assert boundary_wo_corner.shape == f_pre_streaming.shape[0:2]
        assert boundary_wo_corner.shape == f_post_streaming.shape[0:2]

        for bound, change_directions in zip([boundary_wo_corner_left, boundary_wo_corner_right],
                                            [change_directions_left, change_directions_right]):
            for change_dir in change_directions:
                f_post_streaming[bound, mapping[change_dir]] = f_pre_streaming[bound, change_dir]

        for corner_index, change_dir in corner_indices:
            if not isinstance(change_dir, tuple):
                change_dir = [change_dir]
            for dir in change_dir:
                f_post_streaming[corner_index[0], corner_index[1], mapping[dir]] = f_pre_streaming[
                    corner_index[0], corner_index[1], dir]

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


def inlet(lattice_grid_shape: Tuple[int, int], density_in: float, velocity_in: float) \
        -> Callable[[np.ndarray], np.ndarray]:
    velocity = np.zeros(lattice_grid_shape + (2,))
    velocity[..., 0] = velocity_in
    f_eq = equilibrium_distr_func(
        np.ones(lattice_grid_shape) * density_in,
        velocity
    )

    def bc(f_post_streaming: np.ndarray) -> np.ndarray:
        for i in range(f_post_streaming.shape[-1]):
            f_post_streaming[0, ..., i] = f_eq[0, ..., i]
        return f_post_streaming

    return bc


def outlet() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    change_directions = [3, 6, 7]

    def bc(f_previous, f_post_streaming: np.ndarray) -> np.ndarray:
        for change_dir in change_directions:
            f_post_streaming[-1, :, change_dir] = f_previous[-2, :, change_dir]
        return f_post_streaming

    return bc


def periodic_with_pressure_variations(boundary: np.ndarray, p_in: float, p_out: float) \
        -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    assert boundary.dtype == 'bool'
    assert np.all(boundary[0, :] == boundary[-1, :]) or np.all(boundary[:, 0] == boundary[:, -1])

    c_s = 1 / np.sqrt(3)
    if np.all(boundary[0, :] == boundary[-1, :]):
        density_in = np.divide(p_in, c_s ** 2)
        density_in = np.ones_like(boundary[0, :]) * density_in
        density_out = np.divide(p_out, c_s ** 2)
        density_out = np.ones_like(boundary[-1, :]) * density_out
        change_directions_1 = [1, 5, 8]  # left to right
        change_directions_2 = [3, 6, 7]  # right to left
    elif np.all(boundary[:, 0] == boundary[:, -1]):
        density_in = np.divide(p_in, c_s ** 2)
        density_in = np.ones_like(boundary[:, 0]) * density_in
        density_out = np.divide(p_out, c_s ** 2)
        density_out = np.ones_like(boundary[:, -1]) * density_out
        change_directions_1 = [2, 5, 6]  # bottom to top
        change_directions_2 = [4, 7, 8]  # top to bottom

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray,
           density: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        assert boundary.shape == f_pre_streaming.shape[0:2]
        assert boundary.shape == f_post_streaming.shape[0:2]

        f_eq = equilibrium_distr_func(density, velocity)
        f_eq_in = equilibrium_distr_func(density_in, velocity[-2, ...]).squeeze()
        f_post_streaming[0, ..., change_directions_1] = f_eq_in[..., change_directions_1].T + (
                f_pre_streaming[-2, ..., change_directions_1] - f_eq[-2, ..., change_directions_1])

        f_eq_out = equilibrium_distr_func(density_out, velocity[1, ...]).squeeze()
        f_post_streaming[-1, ..., change_directions_2] = f_eq_out[..., change_directions_2].T + (
                f_pre_streaming[1, ..., change_directions_2] - f_eq[1, ..., change_directions_2])

        return f_post_streaming

    return bc
