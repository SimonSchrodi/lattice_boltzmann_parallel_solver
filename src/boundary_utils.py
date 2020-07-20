import numpy as np

from boundary_conditions import rigid_wall, moving_wall, periodic_with_pressure_variations, inlet, outlet

from parallelization_utils import x_in_process, y_in_process, global_to_local_direction


def couette_flow_boundary_conditions(lx: int, ly: int, U: float):
    def boundary(f_pre_streaming, f_post_streaming, density, velocity, f=None):
        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, -1] = np.ones(lx)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)
        boundary_moving_wall = np.zeros((lx, ly))
        boundary_moving_wall[:, 0] = np.ones(lx)
        u_w = np.array(
            [U, 0]
        )
        f_post_streaming = moving_wall(boundary_moving_wall.astype(np.bool), u_w, density)(f_pre_streaming,
                                                                                           f_post_streaming)
        return f_post_streaming

    return boundary


def poiseuille_flow_boundary_conditions(lx: int, ly: int, p_in: float, p_out: float):
    def boundary(f_pre_streaming, f_post_streaming, density, velocity, f=None):
        boundary = np.zeros((lx, ly))
        boundary[0, :] = np.ones(ly)
        boundary[-1, :] = np.ones(ly)
        f_post_streaming = periodic_with_pressure_variations(boundary.astype(np.bool), p_in, p_out)(
            f_pre_streaming,
            f_post_streaming,
            density, velocity)

        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, 0] = np.ones(lx)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)
        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, -1] = np.ones(lx)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)

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
                                            plate_size: int):
    def bc(f_pre_streaming, f_post_streaming, density=None, velocity=None, f_previous=None):
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
        y_min, y_max = ly // 2 - plate_size // 2 + 1, ly // 2 + plate_size // 2 - 1
        if x_in_process(coord2d, lx // 4, lx, x_size):  # left side
            local_x = global_to_local_direction(coord2d[0], lx // 4, lx, x_size)
            for y in range(y_min, y_max):
                if y_in_process(coord2d, y, ly, y_size):
                    local_y = global_to_local_direction(coord2d[1], y, ly, y_size)
                    f_post_streaming[local_x, local_y, [3, 7, 6]] = f_pre_streaming[local_x, local_y, [1, 5, 8]]

            if y_in_process(coord2d, ly // 2 + plate_size // 2 - 1, ly, y_size):  # left side upper corner
                local_y = global_to_local_direction(coord2d[1], ly // 2 + plate_size // 2 - 1, ly, y_size)
                f_post_streaming[local_x, local_y, [3, 6]] = f_pre_streaming[local_x, local_y, [1, 8]]
            if y_in_process(coord2d, ly // 2 - plate_size // 2, ly, y_size):  # left side lower corner
                local_y = global_to_local_direction(coord2d[1], ly // 2 - plate_size // 2, ly, y_size)
                f_post_streaming[local_x, local_y, [3, 7]] = f_pre_streaming[local_x, local_y, [1, 5]]

        if x_in_process(coord2d, lx // 4 + 1, lx, x_size):  # right side
            local_x = global_to_local_direction(coord2d[0], lx // 4 + 1, lx, x_size)
            for y in range(y_min, y_max):
                if y_in_process(coord2d, y, ly, y_size):
                    local_y = global_to_local_direction(coord2d[1], y, ly, y_size)
                    f_post_streaming[local_x, local_y, [1, 5, 8]] = f_pre_streaming[local_x, local_y, [3, 7, 6]]

            if y_in_process(coord2d, ly // 2 + plate_size // 2 - 1, ly, y_size):  # right side upper corner
                local_y = global_to_local_direction(coord2d[1], ly // 2 + plate_size // 2 - 1, ly, y_size)
                f_post_streaming[local_x, local_y, [1, 5]] = f_pre_streaming[local_x, local_y, [3, 7]]
            if y_in_process(coord2d, ly // 2 - plate_size // 2, ly, y_size):  # right side lower corner
                local_y = global_to_local_direction(coord2d[1], ly // 2 - plate_size // 2, ly, y_size)
                f_post_streaming[local_x, local_y, [1, 8]] = f_pre_streaming[local_x, local_y, [3, 6]]

        return f_post_streaming

    return bc
