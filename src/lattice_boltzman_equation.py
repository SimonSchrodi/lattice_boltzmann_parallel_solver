import numpy as np
import typing


def compute_density(prob_densitiy_func: np.ndarray):
    assert prob_densitiy_func.shape[-1] == 9
    return np.sum(prob_densitiy_func, axis=2)


def compute_velocity_field(density_func: np.ndarray, prob_density_func: np.ndarray):
    assert prob_density_func.shape[-1] == 9

    velocity_set = np.array(
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

    return np.divide(
        prob_density_func @ velocity_set,
        density_func[..., np.newaxis]
    )


def get_d2q9_constants():
    w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
    return w_i


def streaming(prob_density_func: np.ndarray):
    assert prob_density_func.shape[-1] == 9

    velocity_set = np.array(
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

    for i in range(prob_density_func.shape[-1]):
        np.roll(prob_density_func[:, :, i], velocity_set[i], axis=(0, 1))

    return prob_density_func
