import unittest
from landscape import landscape_gen
import numpy as np

class TestLandscape(unittest.TestCase):
    def setUp(self):
        self.landscape = landscape_gen(lat = 200, long = 200)
        self.landscape.centroids = np.asarray([[  0.         ,200.        ], [200.           ,0.        ], [200.         ,200.        ], [  0.           ,0.        ], [  0.         ,100.        ], [100.        ,   0.        ], [100.         ,200.        ], [200.         ,100.        ],[164.74198475  ,24.05747735], [132.11272711 ,108.1256016 ], [ 94.63720828  ,68.80952352], [174.38847647 ,116.60099788],[ 66.47637797 ,139.18754889]])
        self.landscape.heights = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.11070644243225414, 0.3668895455363128, -0.28487031743460167, 0.17991270123632352, -0.007167784173236491])

    def test_get_height(self):
        X, Y = np.meshgrid(np.linspace(0, 200, 1024), np.linspace(0, 200, 1024))
        Z = self.landscape.get_height(X, Y)
        self.assertIsInstance(Z, np.ndarray)
        self.assertIsNotNone(Z, "Z is None")
        self.assertEqual(Z.shape, (1024, 1024), "Z shape is not (1024, 1024)")


