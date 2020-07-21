import numpy as np
from typing import Callable

from boundary_conditions import rigid_wall, moving_wall, periodic_with_pressure_variations, inlet, outlet

from parallelization_utils import x_in_process, y_in_process, global_to_local_direction


def couette_flow_boundary_conditions(lx: int, ly: int, U: float, avg_density: float) \
        -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    boundary_rigid_wall = np.zeros((lx, ly))
    boundary_rigid_wall[:, -1] = np.ones(lx)
    top_wall_func = rigid_wall(boundary_rigid_wall.astype(np.bool))
    boundary_moving_wall = np.zeros((lx, ly))
    boundary_moving_wall[:, 0] = np.ones(lx)
    u_w = np.array(
        [U, 0]
    )
    moving_wall_func = moving_wall(boundary_moving_wall.astype(np.bool), u_w, avg_density)

    def boundary(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray, density: np.ndarray = None,
                 velocity: np.ndarray = None, f: np.ndarray = None) -> np.ndarray:
        f_post_streaming = top_wall_func(f_pre_streaming, f_post_streaming)
        f_post_streaming = moving_wall_func(f_pre_streaming, f_post_streaming)
        return f_post_streaming

    return boundary


def poiseuille_flow_boundary_conditions(lx: int, ly: int, p_in: float, p_out: float) \
        -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    boundary = np.zeros((lx, ly))
    boundary[0, :] = np.ones(ly)
    boundary[-1, :] = np.ones(ly)
    periodic_with_pressure_variations_func = periodic_with_pressure_variations(boundary.astype(np.bool), p_in, p_out)

    boundary_bottom_rigid_wall = np.zeros((lx, ly))
    boundary_bottom_rigid_wall[:, 0] = np.ones(lx)
    bottom_rigid_wall_func = rigid_wall(boundary_bottom_rigid_wall.astype(np.bool))

    boundary_top_rigid_wall = np.zeros((lx, ly))
    boundary_top_rigid_wall[:, -1] = np.ones(lx)
    top_rigid_wall_func = rigid_wall(boundary_top_rigid_wall.astype(np.bool))

    def boundary(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray, density: np.ndarray, velocity: np.ndarray,
                 f: np.ndarray = None) -> np.ndarray:
        f_post_streaming = periodic_with_pressure_variations_func(f_pre_streaming, f_post_streaming, density, velocity)
        f_post_streaming = bottom_rigid_wall_func(f_pre_streaming, f_post_streaming)
        f_post_streaming = top_rigid_wall_func(f_pre_streaming, f_post_streaming)

        return f_post_streaming

    return boundary


def parallel_von_karman_boundary_conditions(coord2d,
                                            n_local_x: int,
                                            n_local_y: int,
                                            lx: int,
                                            ly: int,
                                            x_size: int,
                                            y_size: int,
                                            density_in: float,
                                            velocity_in: float,
                                            plate_size: int) \
        -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    y_min, y_max = ly // 2 - plate_size // 2 + 1, ly // 2 + plate_size // 2 - 1
    if x_in_process(coord2d, lx // 4, lx, x_size):  # left side
        plate_boundary_left = np.zeros((n_local_x, n_local_y))
        local_x = global_to_local_direction(coord2d[0], lx // 4, lx, x_size)
        for y in range(y_min, y_max):
            if y_in_process(coord2d, y, ly, y_size):
                local_y = global_to_local_direction(coord2d[1], y, ly, y_size)
                plate_boundary_left[local_x, local_y] = 1
        plate_boundary_left = plate_boundary_left.astype(np.bool)

    if x_in_process(coord2d, lx // 4 + 1, lx, x_size):  # right side
        plate_boundary_right = np.zeros((n_local_x, n_local_y))
        local_x = global_to_local_direction(coord2d[0], lx // 4 + 1, lx, x_size)
        for y in range(y_min, y_max):
            if y_in_process(coord2d, y, ly, y_size):
                local_y = global_to_local_direction(coord2d[1], y, ly, y_size)
                plate_boundary_right[local_x, local_y] = 1
        plate_boundary_right = plate_boundary_right.astype(np.bool)

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray, density: np.ndarray = None,
           velocity: np.ndarray = None, f_previous: np.ndarray = None) -> np.ndarray:
        # inlet
        if x_in_process(coord2d, 0, lx, x_size):
            f_post_streaming[1:-1, 1:-1, :] = inlet((n_local_x, n_local_y), density_in, velocity_in)(
                f_post_streaming.copy()[1:-1, 1:-1, :])

        # outlet
        if x_in_process(coord2d, lx - 1, lx, x_size) and x_in_process(coord2d, lx - 2, lx, x_size):
            f_post_streaming[1:-1, 1:-1, :] = outlet()(f_previous.copy()[1:-1, 1:-1, :],
                                                       f_post_streaming.copy()[1:-1, 1:-1, :])
        elif x_in_process(coord2d, lx - 1, lx, x_size) or x_in_process(coord2d, lx - 2, lx, x_size):
            # TODO communicate f_previous
            raise NotImplementedError

        # plate boundary condition
        if x_in_process(coord2d, lx // 4, lx, x_size):  # left side
            f_post_streaming[plate_boundary_left, [3, 7, 6]] = f_pre_streaming[plate_boundary_left, [1, 5, 8]]

            local_x = global_to_local_direction(coord2d[0], lx // 4, lx, x_size)
            if y_in_process(coord2d, ly // 2 + plate_size // 2 - 1, ly, y_size):  # left side upper corner
                local_y = global_to_local_direction(coord2d[1], ly // 2 + plate_size // 2 - 1, ly, y_size)
                f_post_streaming[local_x, local_y, [3, 6]] = f_pre_streaming[local_x, local_y, [1, 8]]
            if y_in_process(coord2d, ly // 2 - plate_size // 2, ly, y_size):  # left side lower corner
                local_y = global_to_local_direction(coord2d[1], ly // 2 - plate_size // 2, ly, y_size)
                f_post_streaming[local_x, local_y, [3, 7]] = f_pre_streaming[local_x, local_y, [1, 5]]

        if x_in_process(coord2d, lx // 4 + 1, lx, x_size):  # right side
            f_post_streaming[plate_boundary_right, [1, 5, 8]] = f_pre_streaming[plate_boundary_right, [3, 7, 6]]

            local_x = global_to_local_direction(coord2d[0], lx // 4 + 1, lx, x_size)
            if y_in_process(coord2d, ly // 2 + plate_size // 2 - 1, ly, y_size):  # right side upper corner
                local_y = global_to_local_direction(coord2d[1], ly // 2 + plate_size // 2 - 1, ly, y_size)
                f_post_streaming[local_x, local_y, [1, 5]] = f_pre_streaming[local_x, local_y, [3, 7]]
            if y_in_process(coord2d, ly // 2 - plate_size // 2, ly, y_size):  # right side lower corner
                local_y = global_to_local_direction(coord2d[1], ly // 2 - plate_size // 2, ly, y_size)
                f_post_streaming[local_x, local_y, [1, 8]] = f_pre_streaming[local_x, local_y, [3, 6]]

        return f_post_streaming

    return bc
