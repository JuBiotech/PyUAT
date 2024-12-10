""" Test tracking models
"""
import unittest

import numpy as np

from uatrack.models import predict_movement, predict_property


class FakeTracking:
    """Make a fake tracking"""

    def __init__(self, tracking_data):
        self.parents = tracking_data


class TestModels(unittest.TestCase):
    """Test contour functionality"""

    @staticmethod
    def test_movement_predict():
        tracking = FakeTracking(np.array([-1, 0, 0, 2]))
        data = {
            "centroid": np.array(
                [
                    [0, 0],
                    [2, 0],
                    [0, 2],
                    [0, 0],
                ]
            )
        }

        # predict movement for one step
        pmovement = predict_movement(
            tracking,
            np.array([1, 2], int).reshape((-1, 1)),
            data,
            1,
            None,
            stop_at_split=False,
        )
        np.testing.assert_almost_equal(pmovement, np.array([[4, 0], [0, 4]]))

        # predict movement for one step
        pmovement = predict_movement(
            tracking,
            np.array([1, 2], int).reshape((-1, 1)),
            data,
            1,
            None,
            stop_at_split=True,
        )
        np.testing.assert_almost_equal(pmovement, np.array([[2, 0], [0, 2]]))

    @staticmethod
    def test_movement_predict_general():
        tracking = FakeTracking(np.array([-1, 0, 0, 2]))
        data = {
            "centroid": np.array(
                [
                    [0, 0],
                    [2, 0],
                    [0, 2],
                    [0, 0],
                ]
            )
        }

        # predict movement for one step
        pmovement = predict_property(
            tracking,
            np.array([1, 2], int).reshape((-1, 1)),
            data,
            1,
            None,
            stop_at_split=False,
            property_name="centroid",
        )
        np.testing.assert_almost_equal(pmovement, np.array([[4, 0], [0, 4]]))

        # predict movement for one step
        pmovement = predict_property(
            tracking,
            np.array([1, 2], int).reshape((-1, 1)),
            data,
            1,
            None,
            stop_at_split=True,
            property_name="centroid",
        )
        np.testing.assert_almost_equal(pmovement, np.array([[2, 0], [0, 2]]))

    @staticmethod
    def test_growth_predict():
        tracking = FakeTracking(np.array([-1, 0, 1, 2]))
        data = {"area": np.array([2, 4, 8, 12])}  # 0  # 1  # 2  # 3

        # predict movement for one step
        pmovement = predict_property(
            tracking,
            np.array([1, 2], int).reshape((-1, 1)),
            data,
            1,
            None,
            property_name="area",
        )
        np.testing.assert_almost_equal(pmovement, np.array([6, 12]))

        pmovement = predict_property(
            tracking,
            np.array([0, 1, 2], int).reshape((-1, 1)),
            data,
            1,
            None,
            property_name="area",
        )
        np.testing.assert_almost_equal(pmovement, np.array([2, 6, 12]))
        # predict movement for one step
        # pmovement = predict_movement(tracking, np.array([1,2], int).reshape((-1, 1)), data, 1, None, stop_at_split=True)
        # np.testing.assert_almost_equal(pmovement, np.array([[2, 0], [0, 2]]))


if __name__ == "__main__":
    unittest.main()
