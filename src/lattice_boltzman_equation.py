import numpy as np
import typing


def get_velocity_sets():
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


def get_w_i():
    """
    Return weights defined for a D2Q9 lattice. The weighting takes the different lengths of
    the discrete velocities of the velocity set into account.
    Returns:
        weights defined for a D2Q9 lattice
    """
    return np.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    )


def compute_density(prob_densitiy_func: np.ndarray):
    """
    Compute local density
    Args:
        prob_densitiy_func: probability density function f(r,v,t)

    Returns:
        local density

    """
    assert prob_densitiy_func.shape[-1] == 9
    return np.sum(prob_densitiy_func, axis=-1)


def compute_velocity_field(density_func: np.ndarray, prob_density_func: np.ndarray):
    """
    Computes the velocity field
    Args:
        density_func: local density rho(x)
        prob_density_func: probability density function f(r,v,t)

    Returns:
        local average velocity

    """
    assert prob_density_func.shape[-1] == 9

    ux = np.divide(
        np.sum(prob_density_func[:, :, [1, 5, 8]], axis=2) - np.sum(prob_density_func[:, :, [3, 6, 7]], axis=2),
        density_func
    )
    uy = np.divide(
        np.sum(prob_density_func[:, :, [2, 5, 6]], axis=2) - np.sum(prob_density_func[:, :, [4, 7, 8]], axis=2),
        density_func
    )

    u = np.dstack((ux, uy))

    return u


def streaming(prob_density_func: np.ndarray):
    """
    Implements the streaming operator
    Args:
        prob_density_func: probability density function f(r,v,t)

    Returns:
        new probability density function

    """
    assert prob_density_func.shape[-1] == 9

    velocity_set = get_velocity_sets()

    new_prob_density_func = np.zeros_like(prob_density_func)
    for i in range(prob_density_func.shape[-1]):
        new_prob_density_func[..., i] = np.roll(prob_density_func[..., i], velocity_set[i], axis=(0, 1))

    return new_prob_density_func


def equilibrium_distr_func(density_func: np.ndarray, velocity_field: np.ndarray, discretized_directions: int):
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
