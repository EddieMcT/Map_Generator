import unittest
import numpy as np
from map_generator.landscape import landscape_gen
from map_generator.imaging_functions import normalize
import os
import cv2

class TestLandscape(unittest.TestCase):
    def setUp(self):
        self.landscape = landscape_gen(lat = 200, long = 200)
        self.landscape.centroids = np.asarray([[0.,200. ], [200.,0.], [200.,200.], [0.,0.], [0.,100.], [100.,0.], [100.,200.], [200.,100.], [164.74198475,24.05747735], [132.11272711,108.1256016], [94.63720828,68.80952352], [174.38847647,116.60099788], [66.47637797,139.18754889]])
        self.landscape.heights = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.11070644243225414, 0.3668895455363128, -0.28487031743460167, 0.17991270123632352, -0.007167784173236491])

    def test_get_height(self):
        for res in [16, 128, 256, 512]:   
            X, Y = np.meshgrid(np.linspace(0, 200, res), np.linspace(0, 200, res))
            base, mountain, Z, _, _ = self.landscape.get_height(X, Y)
            for arr in [base, mountain, Z]:
                self.assertIsInstance(arr, np.ndarray)
                self.assertIsNotNone(arr, "Array is None")
                self.assertEqual(arr.shape, (res, res), f"Array shape is not ({res}, {res})")

    

    def test_load_map(self): #load the created test image, and compare to the golden record for human check
            
        def test_normalize(res = 128): #Included here as it is tied to the output of landscape_gen
            X, Y = np.meshgrid(np.linspace(0, 200, res), np.linspace(0, 200, res))
            base, mountain, Z, _, _ = self.landscape.get_height(X, Y)
            Z = normalize(Z, "tests/current_test.png")
            self.assertIsInstance(Z, np.ndarray)
            self.assertIsNotNone(Z, "Z is None")
            self.assertEqual(Z.shape, (res, res), f"Z shape is not ({res}, {res})")
        
        test_folder = "tests"
        test_normalize(1024)
        current_test_path = os.path.join(test_folder,"current_test.png")
        golden_record_path = os.path.join(test_folder,"golden_record.png")
        self.assertTrue(os.path.exists(current_test_path), "current_test.png does not exist")
        current_test_img = cv2.imread(current_test_path)
        self.assertIsNotNone(current_test_img, "Failed to load current_test.png")

        if os.path.exists(golden_record_path):
            golden_record_img = cv2.imread(golden_record_path)

            # Check if the heights match
            if current_test_img.shape[0] != golden_record_img.shape[0]:
                # Calculate scaling factor to match the height
                scale_factor = current_test_img.shape[0] / golden_record_img.shape[0]
                new_width = int(golden_record_img.shape[1] * scale_factor)
                new_height = current_test_img.shape[0]
                
                # Resize golden_record to match the height of current_test
                golden_record_img = cv2.resize(golden_record_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            side_by_side_img = cv2.hconcat([current_test_img, golden_record_img])
            cv2.imshow("current_test vs golden_record", side_by_side_img)
        else:
            cv2.imshow("current_test", current_test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()


