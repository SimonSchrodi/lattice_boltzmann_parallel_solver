import numpy as np
from typing import Callable, Tuple
import mpi4py


def communication(comm: mpi4py.MPI.Intracomm, left_dst: int, right_dst: int, bottom_dst: int, top_dst: int) \
        -> Callable[[np.ndarray], np.ndarray]:
    def communicate(f: np.ndarray) -> np.ndarray:
        # send to left
        recvbuf = f[-1, ...].copy()
        comm.Sendrecv(f[1, ...].copy(), left_dst, recvbuf=recvbuf, source=right_dst)
        f[-1, ...] = recvbuf
        # send to right
        recvbuf = f[0, ...].copy()
        comm.Sendrecv(f[-2, ...].copy(), right_dst, recvbuf=recvbuf, source=left_dst)
        f[0, ...] = recvbuf
        # send to top
        recvbuf = f[:, -1, :].copy()
        comm.Sendrecv(f[:, 1, :].copy(), bottom_dst, recvbuf=recvbuf, source=top_dst)
        f[:, -1, :] = recvbuf
        # send to bottom
        recvbuf = f[:, 0, :].copy()
        comm.Sendrecv(f[:, -2, :].copy(), top_dst, recvbuf=recvbuf, source=bottom_dst)
        f[:, 0, :] = recvbuf

    return communicate


def get_local_coords(coords2d: list, lx: int, ly: int, x_size: int, y_size: int) -> Tuple[int, int]:
    n_local_x = lx // x_size
    n_local_y = ly // y_size
    if coords2d[0] + 1 <= lx - n_local_x * x_size:
        n_local_x += 1

    if coords2d[1] + 1 <= ly - n_local_y * y_size:
        n_local_y += 1

    return int(n_local_x), int(n_local_y)


def global_to_local_direction(coord1d: list, global_dir: int, lattice_dir: int, dir_size: int):
    return global_dir - coord1d * (lattice_dir // dir_size)


def global_coord_to_local_coord(coord2d: list, global_x: int, global_y: int, lx: int, ly: int, x_size: int,
                                y_size: int) -> Tuple[int, int]:
    if x_in_process(coord2d, global_x, lx, x_size) and y_in_process(coord2d, global_y, ly, y_size):
        local_x = global_to_local_direction(coord2d[0], global_x, lx, x_size)
        local_y = global_to_local_direction(coord2d[1], global_y, ly, y_size)
        return coord2d, int(local_x), int(local_y)
    return None, None, None


def x_in_process(coord2d: list, x_coord: int, lx: int, processes_in_x: int) -> bool:
    lower = coord2d[0] * (lx // processes_in_x)
    upper = (coord2d[0] + 1) * (lx // processes_in_x) - 1 if not coord2d[0] == processes_in_x - 1 else lx
    return lower <= x_coord <= upper


def y_in_process(coord2d: list, y_coord: int, ly: int, processes_in_y: int) -> bool:
    lower = coord2d[1] * (ly // processes_in_y)
    upper = (coord2d[1] + 1) * (ly // processes_in_y) - 1 if not coord2d[1] == processes_in_y - 1 else ly
    return lower <= y_coord <= upper
