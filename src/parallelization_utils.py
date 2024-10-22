import numpy as np
from typing import Callable, Tuple
from mpi4py import MPI


def communication(comm: MPI.Intracomm) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrapper for communcation step function

    Args:
        comm: MPI communicator

    Returns:
        function which implements the communication step

    """
    # precompute
    left_src, left_dst = comm.Shift(direction=0, disp=-1)
    right_src, right_dst = comm.Shift(direction=0, disp=1)
    bottom_src, bottom_dst = comm.Shift(direction=1, disp=-1)
    top_src, top_dst = comm.Shift(direction=1, disp=1)

    def communicate(f: np.ndarray) -> np.ndarray:
        """
        Implements the communication step. For details we refer to the report in the repository.

        Args:
            f: probability density function

        Returns:
            probability density function after communication

        """
        # send to left
        recvbuf = f[-1, ...].copy()
        comm.Sendrecv(f[1, ...].copy(), left_dst, recvbuf=recvbuf, source=left_src)
        f[-1, ...] = recvbuf
        # send to right
        recvbuf = f[0, ...].copy()
        comm.Sendrecv(f[-2, ...].copy(), right_dst, recvbuf=recvbuf, source=right_src)
        f[0, ...] = recvbuf
        # send to bottom
        recvbuf = f[:, -1, :].copy()
        comm.Sendrecv(f[:, 1, :].copy(), bottom_dst, recvbuf=recvbuf, source=bottom_src)
        f[:, -1, :] = recvbuf
        # send to top
        recvbuf = f[:, 0, :].copy()
        comm.Sendrecv(f[:, -2, :].copy(), top_dst, recvbuf=recvbuf, source=top_src)
        f[:, 0, :] = recvbuf
        return f

    return communicate


def get_xy_size(total_number_of_procces: int) -> Tuple[int, int]:
    """
    Computes how the processes align in the cartesian topology. We try to arange them in a order s.t. they are almost
    quadratic given a quadratic computational domain.

    Args:
        total_number_of_procces: total number of processes

    Returns:
        number of processes in x direction, number of processes in x direction

    """
    def is_prime(x: int) -> bool:
        if x <= 2:
            return False
        for i in range(2, x):
            if (x % i) == 0:
                return False
        return True

    if not (total_number_of_procces == 1) and not (total_number_of_procces == 2) and is_prime(total_number_of_procces):
        raise Exception('This implementation does not work if number of nodes is a prime (excluding 1 and 2)')

    if total_number_of_procces > 1:
        # search for best cartesian topology
        square_root = np.sqrt(total_number_of_procces)
        lower = np.ceil(square_root)
        upper = np.ceil(square_root)

        while lower * upper != total_number_of_procces:
            if lower * upper > total_number_of_procces:
                lower -= 1
            elif lower * upper < total_number_of_procces:
                upper += 1

        return lower, upper

    return 1, 1


def get_local_coords(coords2d: list, lx: int, ly: int, x_size: int, y_size: int) -> Tuple[int, int]:
    """
    Computes the grid sizes for each process. If lx//x_size or ly//y_size are not integers, we extend the last process
    in the respective direction, e.g. the rightmost or topmost processes.

    Args:
        coords2d: process coordinates given the cartesian topology
        lx: lattice grid size in x direction
        ly: lattice grid size in y direction
        x_size: number of process in x direction
        y_size: number of process in y direction

    Returns:
        local grid size in x direction, local grid size in y direction

    """
    n_local_x = lx // x_size
    n_local_y = ly // y_size

    if coords2d[0] + 1 == x_size:
        n_local_x = lx - n_local_x * (x_size - 1)

    if coords2d[1] + 1 == y_size:
        n_local_y = ly - n_local_y * (y_size - 1)

    return int(n_local_x), int(n_local_y)


def global_to_local_direction(coord1d: int, global_dir: int, lattice_dir: int, dir_size: int):
    """
    Helper function to compute local coordinate

    Args:
        coord1d: process coordinate in the x or y
        global_dir: global coordinate in x or y
        lattice_dir: lattice grid size in x or y
        dir_size: number of processes in x or y given the cartesian topology

    Returns:
        local x or y coordinate

    """
    return int(global_dir - coord1d * (lattice_dir // dir_size)) + 1  # +1 due to ghost cell


def global_coord_to_local_coord(coord2d: list, global_x: int, global_y: int, lx: int, ly: int, x_size: int,
                                y_size: int) -> Tuple[int, int]:
    """
    Transforms a global coordinate into a local coordinate

    Args:
        coord2d: process coordinates given the cartesian topology
        global_x: global x coordinate
        global_y: global x coordinate
        lx: lattice grid size in x direction
        ly: lattice grid size in y direction
        x_size: number of processes in x direction
        y_size: number of processes in y direction

    Returns:
        process coordinates, local x coordinate and local y coordinate

    """
    if x_in_process(coord2d, global_x, lx, x_size) and y_in_process(coord2d, global_y, ly, y_size):
        local_x = global_to_local_direction(coord2d[0], global_x, lx, x_size)
        local_y = global_to_local_direction(coord2d[1], global_y, ly, y_size)
        return coord2d, local_x, local_y
    return None, None, None


def x_in_process(coord2d: list, x_coord: int, lx: int, processes_in_x: int) -> bool:
    """
    Checks whether a global x coordinate is in a given process or not.

    Args:
        coord2d: process coordinates given the cartesian topology
        x_coord: global x coordinate
        lx: size of the lattice grid in x direction
        processes_in_x: number of processes in x direction

    Returns:
        boolean whether global x coordinate is in given process or not

    """
    lower = coord2d[0] * (lx // processes_in_x)
    upper = (coord2d[0] + 1) * (lx // processes_in_x) - 1 if not coord2d[0] == processes_in_x - 1 else lx - 1
    return lower <= x_coord <= upper


def y_in_process(coord2d: list, y_coord: int, ly: int, processes_in_y: int) -> bool:
    """
    Checks whether a global y coordinate is in a given process or not.

    Args:
        coord2d: process coordinates given the cartesian topology
        x_coord: global y coordinate
        lx: size of the lattice grid in y direction
        processes_in_x: number of processes in y direction

    Returns:
        boolean whether global y coordinate is in given process or not

    """
    lower = coord2d[1] * (ly // processes_in_y)
    upper = (coord2d[1] + 1) * (ly // processes_in_y) - 1 if not coord2d[1] == processes_in_y - 1 else ly - 1
    return lower <= y_coord <= upper


def save_mpiio(comm: MPI.Intracomm, fn: str, g_kl: np.ndarray):
    """
    Write a global two-dimensional array to a single file in the npy format
    using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

    Arrays written with this function can be read with numpy.load.

    Args:
        comm: MPI communicator
        fn: File name
        g_kl: array_like Portion of the array on this MPI processes. This needs to be a two-dimensional array.

    """
    from numpy.lib.format import dtype_to_descr, magic
    magic_str = magic(1, 0)

    local_nx, local_ny = g_kl.shape
    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)

    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({'descr': dtype_to_descr(g_kl.dtype),
                        'fortran_order': False,
                        'shape': (np.asscalar(nx), np.asscalar(ny))})
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny * local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety + offsetx) * mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()
