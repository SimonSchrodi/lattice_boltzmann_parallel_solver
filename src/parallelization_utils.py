import numpy as np
from typing import Callable


def communication(comm, left_dst, right_dst, bottom_dst, top_dst) -> Callable[[np.ndarray], np.ndarray]:
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


def x_in_process(coord3d, x_coord: int, lx: int, processes_in_x: int):
    lower = coord3d[0] * (lx // processes_in_x)
    upper = (coord3d[0] + 1) * (lx // processes_in_x) - 1 if not coord3d[0] == processes_in_x - 1 else lx
    return lower <= x_coord <= upper


def y_in_process(coord3d, y_coord: int, ly: int, processes_in_y: int):
    lower = coord3d[1] * (ly // processes_in_y)
    upper = (coord3d[1] + 1) * (ly // processes_in_y) - 1 if not coord3d[1] == processes_in_y - 1 else ly
    return lower <= y_coord <= upper
