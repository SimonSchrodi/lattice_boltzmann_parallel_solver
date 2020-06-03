import unittest
import logging
import numpy as np

from src.lattice_boltzman_equation import streaming


class TestMassPreservation(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_prob_density_func_constant(self):
        """
        Test mass preservation of streaming function
        """
        prob_density_func = np.ones((10, 10, 9))
        new_prob_density_func = streaming(prob_density_func)
        self.assertEqual(prob_density_func.shape, new_prob_density_func.shape)
        self.assertAlmostEqual(np.sum(prob_density_func), np.sum(new_prob_density_func), places=1)

    def test_prob_density_func_random(self):
        """
        Test mass preservation of streaming function
        """
        prob_density_func = np.random.random((10, 10, 9))
        new_prob_density_func = streaming(prob_density_func)
        self.assertEqual(prob_density_func.shape, new_prob_density_func.shape)
        self.assertAlmostEqual(np.sum(prob_density_func), np.sum(new_prob_density_func), places=1)

    def test_prob_density_func_just_one_dimension(self):
        """
        Test mass preservation of streaming function
        """
        prob_density_func = np.zeros((10, 10, 9))
        prob_density_func[..., 1] = np.ones((10, 10))
        new_prob_density_func = streaming(prob_density_func)
        self.assertEqual(prob_density_func.shape, new_prob_density_func.shape)
        self.assertAlmostEqual(np.sum(prob_density_func), np.sum(new_prob_density_func), places=1)


if __name__ == "__main__":
    unittest.main()