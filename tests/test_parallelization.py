import numpy as np
from mpi4py import MPI
import os
import shutil

from tqdm import tqdm

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './src')

from parallelization_utils import x_in_process, y_in_process, save_mpiio, global_to_local_direction, get_local_coords, global_coord_to_local_coord, communication, get_xy_size

from boundary_conditions import inlet, outlet
from lattice_boltzman_equation import equilibrium_distr_func, lattice_boltzman_step
from initial_values import density_1_velocity_x_u0_velocity_y_0_initial

def run_test():
    lx, ly = 420, 180
    d = 40
    u0 = 0.1
    density_in = 1.0
    kinematic_viscosity = 0.04
    omega = np.reciprocal(3 * kinematic_viscosity + 0.5)
    time_steps = 11

    p_coords = [3 * lx // 4, ly // 2]

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
    x_size, y_size = get_xy_size(size)

    cartesian2d = comm.Create_cart(dims=[x_size, y_size], periods=[True, True], reorder=False)
    coords2d = cartesian2d.Get_coords(rank)

    n_local_x, n_local_y = get_local_coords(coords2d, lx, ly, x_size, y_size)

    def boundary(coord2d, n_local_x, n_local_y):
        def bc(f_pre_streaming, f_post_streaming, density=None, velocity=None, f_previous=None):
            # inlet
            if x_in_process(coord2d, 0, lx, x_size):
                f_post_streaming[1:-1, 1:-1, :] = inlet((n_local_x, n_local_y), density_in, u0)(
                    f_post_streaming.copy()[1:-1, 1:-1, :])

            # outlet
            if x_in_process(coord2d, lx - 1, lx, x_size) and x_in_process(coord2d, lx - 2, lx, x_size):
                f_post_streaming[1:-1, 1:-1, :] = outlet()(f_previous.copy()[1:-1, 1:-1, :],
                                                           f_post_streaming.copy()[1:-1, 1:-1, :])
            elif x_in_process(coord2d, lx - 1, lx, x_size) or x_in_process(coord2d, lx - 2, lx, x_size):
                # TODO communicate f_previous
                raise NotImplementedError

            # plate boundary condition
            y_min, y_max = ly // 2 - d // 2 + 1, ly // 2 + d // 2 - 1
            if x_in_process(coord2d, lx // 4, lx, x_size):  # left side
                local_x = global_to_local_direction(coord2d[0], lx // 4, lx, x_size)
                for y in range(y_min, y_max):
                    if y_in_process(coord2d, y, ly, y_size):
                        local_y = global_to_local_direction(coord2d[1], y, ly, y_size)
                        f_post_streaming[local_x, local_y, [3, 7, 6]] = f_pre_streaming[local_x, local_y, [1, 5, 8]]

                if y_in_process(coord2d, ly // 2 + d // 2 - 1, ly, y_size):  # left side upper corner
                    local_y = global_to_local_direction(coord2d[1], ly // 2 + d // 2 - 1, ly, y_size)
                    f_post_streaming[local_x, local_y, [3, 6]] = f_pre_streaming[local_x, local_y, [1, 8]]
                if y_in_process(coord2d, ly // 2 - d // 2, ly, y_size):  # left side lower corner
                    local_y = global_to_local_direction(coord2d[1], ly // 2 - d // 2, ly, y_size)
                    f_post_streaming[local_x, local_y, [3, 7]] = f_pre_streaming[local_x, local_y, [1, 5]]

            if x_in_process(coord2d, lx // 4 + 1, lx, x_size):  # right side
                local_x = global_to_local_direction(coord2d[0], lx // 4 + 1, lx, x_size)
                for y in range(y_min, y_max):
                    if y_in_process(coord2d, y, ly, y_size):
                        local_y = global_to_local_direction(coord2d[1], y, ly, y_size)
                        f_post_streaming[local_x, local_y, [1, 5, 8]] = f_pre_streaming[local_x, local_y, [3, 7, 6]]

                if y_in_process(coord2d, ly // 2 + d // 2 - 1, ly, y_size):  # right side upper corner
                    local_y = global_to_local_direction(coord2d[1], ly // 2 + d // 2 - 1, ly, y_size)
                    f_post_streaming[local_x, local_y, [1, 5]] = f_pre_streaming[local_x, local_y, [3, 7]]
                if y_in_process(coord2d, ly // 2 - d // 2, ly, y_size):  # right side lower corner
                    local_y = global_to_local_direction(coord2d[1], ly // 2 - d // 2, ly, y_size)
                    f_post_streaming[local_x, local_y, [1, 8]] = f_pre_streaming[local_x, local_y, [3, 6]]

            return f_post_streaming

        return bc

    density, velocity = density_1_velocity_x_u0_velocity_y_0_initial((n_local_x + 2, n_local_y + 2), u0)
    f = equilibrium_distr_func(density, velocity)
    process_coord, px, py = global_coord_to_local_coord(coords2d, p_coords[0], p_coords[1], lx, ly, x_size, y_size)
    if process_coord is not None:
        vel_at_p = [np.linalg.norm(velocity[px, py, ...])]

    bound_func = boundary(coords2d, n_local_x, n_local_y)
    communication_func = communication(cartesian2d)

    if rank == 0:
        pbar = tqdm(total=time_steps)
    for i in range(time_steps):
        if rank == 0:
            pbar.update(1)
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega, bound_func, communication_func)
        if process_coord is not None:
            vel_at_p.append(np.linalg.norm(velocity[px, py, ...]))
            vel_at_p_test = np.load(r'./tests/von_karman_vortex_shedding/vel_at_p.npy')
            assert vel_at_p[-1] == vel_at_p_test[i + 1]

        for j in range(9):
            save_mpiio(cartesian2d, r'./tests/tmp/f_' + str(j) + '.npy', f[1:-1, 1:-1, j])
        if rank == 0:
            f_gather = [np.load(r'./tests/tmp/f_' + str(j) + '.npy') for j in range(9)]
            f_gather = np.stack(f_gather, axis=-1)
            f_test = np.load(r'./tests/von_karman_vortex_shedding/f_' + str(i) + '.npy')
            assert f_gather.shape == f_test.shape
            print(np.unique(f_gather == f_test, return_counts=True))
            assert np.allclose(f_gather, f_test)

if __name__ == "__main__":
    if not os.path.exists(r'./tests/tmp'):
        os.makedirs(r'./tests/tmp')
    run_test()
    shutil.rmtree(r'./tests/tmp')