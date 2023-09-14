#  This function tests the distance function in the distance.py file

import unittest
import cobsurv.models.distance_functions as distance_functions
import numpy as np

class TestDistance(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(distance_functions.distance_trapz(np.array([2,2,4]),np.array([1,3,1]),np.array([6.7,8.2,9.0])),1.3478255009454345)
        self.assertEqual(distance_functions.distance_max(np.array([2,2,4]),np.array([1,3,1]),np.array([6.7,8.2,9.0])),3)
        self.assertEqual(distance_functions.distance_euclidean(np.array([2,2,4]),np.array([1,3,1]),np.array([6.7,8.2,9.0])),3.3166247903554)
        self.assertEqual(distance_functions.distance_euler(np.array([2,2,4]),np.array([1,3,1]),np.array([6.7,8.2,9.0])),4.608693648394067)




if __name__ == '__main__':
    unittest.main()
