import numpy as np
from typing import Tuple, Callable


def get_velocity_sets() -> np.ndarray:
    """
    Get velocity set. Note that the length of the discrete velocities can be different.

    Returns:
        velocity set

    """

    return np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]
    )


def vel_to_opp_vel_mapping() -> np.ndarray:
    """
    Opposite direction

    Returns:
        array with opposite directions of D2Q9

    """
    return np.array(
        [0, 3, 4, 1, 2, 7, 8, 5, 6]
    )


def get_w_i() -> np.ndarray:
    """
    Return weights defined for a D2Q9 lattice. The weighting takes the different lengths of
    the discrete velocities of the velocity set into account.

    Returns:
        weights defined for a D2Q9 lattice
    """
    return np.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    )


def reynolds_number(L: int, u: float, v: float) -> float:
    """
    Compute the Reynolds number

    Args:
        L: characteristic length
        u: flow velocity
        v: kinematic viscosity

    Returns:
        Reynolds number

    """
    return np.divide(
        L * u,
        v
    )


def strouhal_number(f: float, L: int, u: float) -> float:
    """
    Computes the Strouhal number

    Args:
        f: vortex frequency
        L: characteristic length
        u: flow velocity

    Returns:
        Strouhal number

    """
    return np.divide(
        f * L,
        u
    )


def compute_density(prob_densitiy_func: np.ndarray) -> np.ndarray:
    """
    Compute local density

    Args:
        prob_densitiy_func: probability density function f(r,v,t)

    Returns:
        density given the probability density function

    """
    assert prob_densitiy_func.shape[-1] == 9
    return np.sum(prob_densitiy_func, axis=-1)


def compute_velocity_field(density_func: np.ndarray, prob_density_func: np.ndarray) -> np.ndarray:
    """
    Computes the velocity field

    Args:
        density_func: density rho(x)
        prob_density_func: probability density function f(r,v,t)

    Returns:
        velocity field given the density and probability density function

    """
    assert prob_density_func.shape[-1] == 9

    ux = np.divide(
        np.sum(prob_density_func[:, :, [1, 5, 8]], axis=2) - np.sum(prob_density_func[:, :, [3, 6, 7]], axis=2),
        density_func,
        out=np.zeros_like(density_func),
        where=density_func != 0
    )
    uy = np.divide(
        np.sum(prob_density_func[:, :, [2, 5, 6]], axis=2) - np.sum(prob_density_func[:, :, [4, 7, 8]], axis=2),
        density_func,
        out=np.zeros_like(density_func),
        where=density_func != 0
    )

    u = np.dstack((ux, uy))

    return u


def streaming(prob_density_func: np.ndarray) -> np.ndarray:
    """
    Implements the streaming operator

    Args:
        prob_density_func: probability density function f(r,v,t)

    Returns:
        new probability density function after streaming

    """
    assert prob_density_func.shape[-1] == 9

    velocity_set = get_velocity_sets()

    new_prob_density_func = np.zeros_like(prob_density_func)
    for i in range(prob_density_func.shape[-1]):
        new_prob_density_func[..., i] = np.roll(prob_density_func[..., i], velocity_set[i], axis=(0, 1))

    return new_prob_density_func


def equilibrium_distr_func(density_func: np.ndarray, velocity_field: np.ndarray) -> np.ndarray:
    """
    Computes the equilibrium distribution function

    Args:
        density_func: density rho(x)
        velocity_field: velocity u(x)

    Returns:
        equilibrium probability density function

    """
    assert density_func.shape == velocity_field.shape[:-1]

    w_i = get_w_i()
    c_i = get_velocity_sets()

    ci_u = velocity_field @ c_i.T

    f_eq = w_i[np.newaxis, np.newaxis, ...] * np.expand_dims(density_func, axis=-1) * (
            1 +
            3 * ci_u +
            (9 / 2) * np.power(ci_u, 2) -
            (3 / 2) * np.expand_dims(np.power(np.linalg.norm(velocity_field, axis=-1), 2), axis=-1)
    )

    return f_eq


def lattice_boltzmann_step(f: np.ndarray, density: np.ndarray, velocity: np.ndarray, omega: float,
                           boundary: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = None,
                           parallel_communication: Callable[[np.ndarray], np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carries out a lattice boltzmann step

    Args:
        f: probability density function f(r,v,t)
        density: density rho(x)
        velocity: velocity u(x)
        omega: relaxation
        boundary: function which executes the boundary conditions
        parallel_communication: function executing the communication step before streaming for parallel implementation

    Returns:
        new probability density function, new density function, new velocity function

    """
    assert f.shape[0:2] == density.shape
    assert f.shape[0:2] == velocity.shape[0:2]
    assert 0 < omega < 2
    f_eq = equilibrium_distr_func(density, velocity)

    f_pre = (f + (f_eq - f) * omega)

    if parallel_communication is not None:
        f_pre = parallel_communication(f_pre)

    f_post = streaming(f_pre)

    if boundary is not None:
        f_post = boundary(f_pre, f_post, density, velocity, f)

    density = compute_density(f_post)
    velocity = compute_velocity_field(density, f_post)

    return f_post, density, velocity
