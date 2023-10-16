"""Utility functions for config generation"""

from functools import partial

import numpy as np
from scipy.stats import binom, halfnorm, norm
from tensor_tree.impl_np import NP_Impl

from uat.models import ModelExecutor, area_growth_computer, split_dist_computer
from uat.utils import ContourDistanceCache

backend = NP_Impl()


def split_pair_filter(
    source_index, target_index, cdc: ContourDistanceCache, df, distance_threshold=1.0
):
    """Make sure that distances for splitting are not too large"""
    # pylint: disable=unused-argument
    if len(target_index) == 0:
        return np.zeros((len(target_index)), dtype=bool)
    mask = cdc.distance(target_index[:, 0], target_index[:, 1]) < distance_threshold

    return mask


def create_split_children_distance_model(
    data, prob=lambda vs: halfnorm.logsf(vs, loc=0, scale=10)
):
    # pylint: disable=unused-argument
    def split_distance_extractor(
        tracking,
        source_index,
        target_index,
        df,
    ):
        """Extracts the distance between the children of proposed assignments

        Args:
            tracking (_type_): tracking lineage
            source_index (_type_): numpy array of sources for assignments
            target_index (_type_): numpy array of targets for assignments
            df (_type_): dictionary with indexed information

        Returns:
            _type_: euclidean distance between the children of the assignments
        """
        major_extents = df["major_extents"]

        distances = np.zeros(len(target_index), dtype=np.float32)

        for i, (a, b) in enumerate(target_index):
            loc_distances = np.linalg.norm(
                major_extents[a][:, None, :] - major_extents[b][None, :, :], axis=-1
            )

            distances[i] = np.min(loc_distances)

        return distances

    split_children_distance_model = ModelExecutor(
        split_distance_extractor,  # extract the distances
        prob,  # compute the probability
        data,
    )

    return split_children_distance_model


def compute_angle_between_children(maj_extA, maj_extB):
    """Computes the angle between the children

    uses the major axes and the poles in order to orient them accordingly.

    The angle is computed in degrees [0...180]
    """

    loc_distances = np.linalg.norm(maj_extA[:, None, :] - maj_extB[None, :, :], axis=-1)

    min_distance = np.min(loc_distances)
    # print(np.array(np.where(min_distance == loc_distances)).flatten())

    # use the directions pointing towads the new tips
    data = np.array(np.where(min_distance == loc_distances))
    # print(data.shape)
    # print(loc_distances)
    a_ind, b_ind = data[:, 0].flatten()
    adir = maj_extA[1 - a_ind] - maj_extA[a_ind]
    bdir = maj_extB[1 - b_ind] - maj_extB[b_ind]
    # print(adir, bdir)

    return (
        np.arccos(np.dot(adir, bdir) / np.linalg.norm(adir) / np.linalg.norm(bdir))
        * 180
        / np.pi
    )


def log_sum(la, lb):
    """log(a + b) = la + log1p(exp(lb - la)) from https://stackoverflow.com/questions/65233445/how-to-calculate-sums-in-log-space-without-underflow"""
    # la = np.log(a)
    # lb = np.log(b)
    return la + np.log1p(np.exp(lb - la))


def prob_angles(angles, loc, scale):

    # compute the difference
    diff = np.abs(loc - angles)

    # compute the probability of more extreme values
    prob = log_sum(
        norm.logcdf(loc - diff, loc=loc, scale=scale),
        norm.logsf(loc + diff, loc=loc, scale=scale),
    )

    # do not use nan values
    prob[np.isnan(prob)] = -30

    return prob


def create_split_children_angle_model(
    data,
    prob=prob_angles
    #    lambda vs: norm.logpdf(
    #        np.abs(np.cos(vs * (2 * np.pi / 360))),
    #        loc=np.cos(0 * 2 * np.pi / 360),
    #        scale=0.1,
    #    ),
):
    # pylint: disable=unused-argument
    def split_angle_extractor(
        tracking,
        source_index,
        target_index,
        df,
    ):
        major_extents = df["major_extents"]

        cos_angles = np.zeros(len(target_index), dtype=np.float32)

        for i, (a, b) in enumerate(target_index):
            # extract both major axes vectors
            majA = major_extents[a]
            majB = major_extents[b]

            # print(majA)
            # print(majB)

            cos_angles[i] = compute_angle_between_children(majA, majB)

        return cos_angles

    split_children_angle_model = ModelExecutor(split_angle_extractor, prob, data)

    return split_children_angle_model


def create_split_rate_model(data):
    def split_rate_comp(values):
        probs = binom.logpmf(values + 1, n=40, p=10 / 40)
        indices = np.where(values == -1)
        probs[indices] = np.log(
            0.5
        )  # 50% split probability when we have no parents available
        print(probs)
        return probs

    # pylint: disable=unused-variable
    split_rate_model = ModelExecutor(
        split_dist_computer,
        # continue_rate_model.model.invert(),
        split_rate_comp,
        data,
    )

    return split_rate_model


def growth_probs(values, mean=1.06, scale=0.3):
    probs = halfnorm.logsf(np.abs(mean - values), scale=scale)
    return probs


def create_growth_model(data, mean, scale):
    return ModelExecutor(
        area_growth_computer,
        # growth_probs,
        partial(growth_probs, mean=mean, scale=scale),
        data,
    )


def distance_to_previous(tracking, source_index, target_index, data):
    # pylint: disable=unused-argument
    all_centroids = data[
        "centroid"
    ]  # np.array(data['centroid'].to_list(), dtype=np.float32)

    sources = all_centroids[source_index]
    targets = all_centroids[target_index]

    distances = np.linalg.norm(sources - targets, axis=-1)

    # if len(distances.shape) == 2:
    # take the sum of distances (only one distance for migration, two distances for division)
    distances = np.sum(distances, axis=-1)

    return distances


def create_continue_keep_position_model(data, prob):
    return ModelExecutor(
        distance_to_previous,
        prob,
        data,
    )


def create_split_movement_model(data, prob):
    return ModelExecutor(
        distance_to_previous,
        prob,
        data,
    )


class SimpleCDC:
    """class to compute contour distances (approximated by center position in favor of speed)"""

    def __init__(self, positions):
        self.positions = positions

    def distance(self, indA, indB):
        return np.linalg.norm(self.positions[indA] - self.positions[indB], axis=1)
