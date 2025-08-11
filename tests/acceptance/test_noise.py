import unittest
import numpy as np
from map_generator.noise_functions import my_perl

class TestNoise(unittest.TestCase):
    def test_sample(self):
        for res in [16, 128, 256, 512]: 
            X, Y = np.meshgrid(np.linspace(0, 10, res), np.linspace(0, 10, res))
            Z = my_perl.sample(X, Y, neg_octaves=4, octaves=-1, ndims=2)
            self.assertIsInstance(Z, np.ndarray)
            self.assertIsNotNone(Z, "Z is None")
            self.assertEqual(Z.shape, (res, res, 2), f"Z shape is not ({res}, {res}, 2)")
            Z = my_perl.sample(X, Y, neg_octaves=0, octaves=5, ndims=3)
            self.assertIsInstance(Z, np.ndarray)
            self.assertIsNotNone(Z, "Z is None")
            self.assertEqual(Z.shape, (res, res, 3), f"Z shape is not ({res}, {res}, 3)")

            Z = my_perl.sample(X, Y, neg_octaves=0, octaves=5, ndims=3, voron=True)
            self.assertIsInstance(Z, np.ndarray)
            self.assertIsNotNone(Z, "Z is None")
            self.assertEqual(Z.shape, (res, res,3), f"Z shape is not ({res}, {res},3)")
        

