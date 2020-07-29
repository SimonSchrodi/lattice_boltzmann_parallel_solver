import numpy as np

from lattice_boltzmann_method import vel_to_opp_vel_mapping, get_velocity_sets, get_w_i, equilibrium_distr_func

from typing import Callable, Tuple


def get_wall_indices(boundary: np.ndarray) -> np.ndarray:
    """
    Helper function to return indices i, which have to be bounced back at the respective wall

    Args:
        boundary: boolean index map indicating boundary nodes

    Returns:
        indicies to bounce back

    """
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
    """
    Helper function to get the corner indicies for the bounce back of an object not at the boundary

    Args:
        boundary: boolean index map indicating boundary nodes

    Returns:
        list of tuples containing the corner indices and the channels i to bounce back

    """
    assert len(boundary.shape) == 2

    indices = np.argwhere(boundary)
    upper_left_corner = (np.amin(indices[:, 0]), np.amax(indices[:, 1]))
    upper_right_corner = (np.amax(indices[:, 0]) + 1, np.amax(indices[:, 1]))
    lower_left_corner = (np.amin(indices[:, 0]), np.amin(indices[:, 1]))
    lower_right_corner = (np.amax(indices[:, 0]) + 1, np.amin(indices[:, 1]))
    return np.array(
        [
            (upper_left_corner, (1, 8)),
            (lower_left_corner, (1, 5)),
            (upper_right_corner, (3, 7)),
            (lower_right_corner, (3, 6))
        ]
    )


def remove_corner_indices_from_boundary(boundary: np.ndarray, corner_indices: np.ndarray) -> np.ndarray:
    """
    Helper function to remover corner indices from the boundary. This is applied for the bounce-back condition
    for an object not at the boundary

    Args:
        boundary: boolean index map indicating boundary nodes
        corner_indices: indices of the corner of the object

    Returns:
        boundary as boolean index map indicating boundary nodes without the corner indices

    """
    assert len(boundary.shape) == 2
    for corner_index, _ in corner_indices:
        boundary[corner_index[0], corner_index[1]] = False
    return boundary


def rigid_wall(boundary: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns function implementing bounce-back boundary condition

    Args:
        boundary: boolean index map indicating boundary nodes

    Returns:
        function implementing bounce-back boundary condition

    """
    assert boundary.dtype == 'bool'
    change_directions = get_wall_indices(boundary)
    mapping = vel_to_opp_vel_mapping()

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray) -> np.ndarray:
        """
        Implements the bounce back boundary condition

        Args:
            f_pre_streaming: before streaming probability density function
            f_post_streaming: after streaming probability density function

        Returns:
            after streaming probability density function on which bounce-back boundary condition was applied

        """
        assert boundary.shape == f_pre_streaming.shape[0:2]
        assert boundary.shape == f_post_streaming.shape[0:2]

        for change_dir in change_directions:
            f_post_streaming[boundary, mapping[change_dir]] = f_pre_streaming[boundary, change_dir]

        return f_post_streaming

    return bc


def rigid_object(boundary: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns function implementing bounce-back condition for an object not at the boundary

    Args:
        boundary: boolean index map indicating boundary nodes

    Returns:
        function which implements the bounce back condition

    """
    assert boundary.dtype == 'bool'

    # precompute required variables s.t. they do not have to be recomputed over and over again
    corner_indices = get_corner_indices(boundary)
    boundary_wo_corner = remove_corner_indices_from_boundary(boundary, corner_indices)
    boundary_wo_corner_left = boundary_wo_corner
    boundary_wo_corner_right = np.roll(boundary_wo_corner, 1, axis=0)
    change_directions_left = [1, 5, 8]
    change_directions_right = [3, 6, 7]
    mapping = vel_to_opp_vel_mapping()

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray) -> np.ndarray:
        """
        Implements the bounce back condition for an object not at the boundary

        Args:
            f_pre_streaming: before streaming probability density function
            f_post_streaming: after streaming probability density function

        Returns:
            after streaming probability density function on which bounce back condition was applied

        """
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


def moving_wall(boundary: np.ndarray, u_w: np.ndarray, avg_density: np.ndarray) \
        -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns function implementing moving wall

    Args:
        boundary: boolean index map indicating boundary nodes
        u_w: velocity of the wall
        avg_density: average density

    Returns: Returns function implementing moving wall

    """
    assert boundary.dtype == 'bool'

    # precompute required variables s.t. they do not have to be recomputed over and over again
    c_s = 1 / np.sqrt(3)
    mapping = vel_to_opp_vel_mapping()
    c_i = get_velocity_sets()
    w_i = get_w_i()
    change_directions = get_wall_indices(boundary)

    def bc(f_pre_streaming: np.ndarray, f_post_streaming: np.ndarray) -> np.ndarray:
        """
        Implements the moving wall

        Args:
            f_pre_streaming: before streaming probability density function
            f_post_streaming: after streaming probability density function

        Returns:
            after streaming probability density function on which moving wall bc was applied

        """
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
    """
    Returns function implementing the inlet

    Args:
        lattice_grid_shape: size of the computational domain
        density_in: inlet density
        velocity_in: inlet velocity

    Returns:
        function implementing the inlet

    """
    # precompute required variables s.t. they do not have to be recomputed over and over again
    velocity = np.zeros(lattice_grid_shape + (2,))
    velocity[..., 0] = velocity_in
    f_eq = equilibrium_distr_func(
        np.ones(lattice_grid_shape) * density_in,
        velocity
    )

    def bc(f_post_streaming: np.ndarray) -> np.ndarray:
        """
        Implements the inlet

        Args:
            f_post_streaming: after streaming probability density function

        Returns:
            after streaming probability density function on which inlet condition was applied

        """
        for i in range(f_post_streaming.shape[-1]):
            f_post_streaming[0, ..., i] = f_eq[0, ..., i]
        return f_post_streaming

    return bc


def outlet() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns function implementing the outlet

    Returns:
        function implementing the outlet

    """
    change_directions = [3, 6, 7]

    def bc(f_previous: np.ndarray, f_post_streaming: np.ndarray) -> np.ndarray:
        """
        Implements the outlet

        Args:
            f_previous: after streaming probability density function of the previous time step
            f_post_streaming: after streaming probability density function

        Returns:
            after streaming probability density function on which outlet condition was applied

        """
        for change_dir in change_directions:
            f_post_streaming[-1, :, change_dir] = f_previous[-2, :, change_dir]
        return f_post_streaming

    return bc


def periodic_with_pressure_variations(boundary: np.ndarray, p_in: float, p_out: float) \
        -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Return function implememting the periodic boundary condition with pressure variation

    Args:
        boundary: boolean index map indicating boundary nodes
        p_in: ingoing density
        p_out: outgoing density

    Returns:
        function implememnting the periodic boundary condition with pressure variation

    """
    assert boundary.dtype == 'bool'
    assert np.all(boundary[0, :] == boundary[-1, :]) or np.all(boundary[:, 0] == boundary[:, -1])

    # precompute required variables s.t. they do not have to be recomputed over and over again
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

    def bc(f_pre_streaming: np.ndarray, density: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Implements the periodic boundary condition with pressure variation. Note that we first apply the boundary
        condition on the before streaming probability density function and then stream.

        Args:
            f_pre_streaming: before streaming probability density function
            density: current density
            velocity: current velocity

        Returns:
            before streaming probability density function on which the periodic boundary condition with
            pressure variation was applied

        """
        assert boundary.shape == f_pre_streaming.shape[0:2]

        f_eq = equilibrium_distr_func(density, velocity)
        f_eq_in = equilibrium_distr_func(density_in, velocity[-2, ...]).squeeze()
        f_pre_streaming[0, ..., change_directions_1] = f_eq_in[..., change_directions_1].T + (
                f_pre_streaming[-2, ..., change_directions_1] - f_eq[-2, ..., change_directions_1])

        f_eq_out = equilibrium_distr_func(density_out, velocity[1, ...]).squeeze()
        f_pre_streaming[-1, ..., change_directions_2] = f_eq_out[..., change_directions_2].T + (
                f_pre_streaming[1, ..., change_directions_2] - f_eq[1, ..., change_directions_2])

        return f_pre_streaming

    return bc
