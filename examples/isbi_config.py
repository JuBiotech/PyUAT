""" Configuration for the tracking model """

import logging
from functools import partial

import numpy as np
from scipy.stats import beta, binom, halfnorm, norm
from tensor_tree.impl_np import NP_Impl

from uat.assignment import (
    SimpleContinueGenerator,
    SimpleEndTrackAssGenerator,
    SimpleNewAssGenerator,
    SimpleSplitGenerator,
)
from uat.models import (
    ConstantModel,
    ModelExecutor,
    area_growth_computer,
    distance_to_pred_computer,
    distance_to_pred_masked,
    split_dist_computer,
)
from uat.utils import ContourDistanceCache, NearestNeighborCache

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


def create_split_children_angle_model(
    data,
    prob=lambda vs: norm.logpdf(
        np.abs(np.cos(vs * (2 * np.pi / 360))),
        loc=np.cos(0 * 2 * np.pi / 360),
        scale=0.1,
    ),
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
            a_vector = major_extents[a][0] - major_extents[a][1]
            b_vector = major_extents[b][0] - major_extents[b][1]

            # compute the angle between them
            cos_angles[i] = (
                np.dot(a_vector, b_vector)
                / np.linalg.norm(a_vector)
                / np.linalg.norm(b_vector)
            )  # -> cosine of the angle

        return cos_angles

    split_children_angle_model = ModelExecutor(split_angle_extractor, prob, data)

    return split_children_angle_model


def create_split_rate_model(data, subsampling):
    def split_rate_comp(values):
        # probs = binom.logpmf(values + 1, n=40, p=10 / 40)
        mean = 70 / subsampling
        scale = 20 / subsampling

        p = 1 - scale / mean
        n = int(mean / p)

        probs = np.log(binom.pmf(values + 1, n=n, p=p))
        indices = np.where(values == -1)
        probs[indices] = np.log(
            0.5
        )  # 50% split probability when we have no parents available
        # print(probs)
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

    return np.linalg.norm(sources - targets, axis=-1).squeeze()


def create_continue_keep_position_model(data, max_distance):
    return ModelExecutor(
        # partial(distance_to_pred_computer, alpha=1, history=0, stop_at_split=True),
        distance_to_previous,
        lambda values: beta.logsf(
            np.clip(values.squeeze() / max_distance, 0, 1), a=1, b=3
        ),
        # lambda values: expon.logsf(values, scale=5),
        data,
    )


def create_split_movement_model(data, max_distance):
    return ModelExecutor(
        partial(distance_to_pred_computer, alpha=1, history=0, stop_at_split=True),
        lambda values: beta.logsf(
            np.clip(values.squeeze() / max_distance, 0, 1), a=1, b=3
        ).sum(axis=-1),
        # lambda values: expon.logsf(values, scale=5).sum(axis=-1),
        data,
    )


def setup_assignment_generators(df, data, width, subsampling):
    # TODO: Why these values?
    constant_new_model = ConstantModel(np.log(0.25**2))
    constant_end_model = ConstantModel(np.log(0.25**2))

    def position(tracking, source_index, target_index, df):
        # pylint: disable=unused-argument
        all_positions = df["centroid"]

        return all_positions[source_index.reshape(-1)]

    def pos_log_pdf(values):
        values = values[:, 0]
        min_dist = np.minimum(values, width - values)

        p = np.zeros_like(values)
        p[min_dist > 60] = -1  # -6 #np.log(1e-6)

        return p

    end_pos_model = ModelExecutor(position, pos_log_pdf, data)

    class SimpleCDC:
        """class to compute contour distances (approximated by center position in favor of speed)"""

        @staticmethod
        def distance(indA, indB):
            return np.linalg.norm(positions[indA] - positions[indB], axis=1)

    cdc = SimpleCDC

    logging.info("Compute nearest neighbor cache")
    nnc = NearestNeighborCache(df)

    positions = data["centroid"]

    filters = [lambda s, t: nnc.kNearestNeigborsMatrixMask(15, s, t)]

    # def sf_debug(s,t):
    #    print(s,t)

    split_filters = []  # [sf_debug]

    max_distance = 200

    continue_keep_position_model = create_continue_keep_position_model(
        data, max_distance
    )

    def movement_log_pdf(values):

        p = halfnorm.logsf(x=values, scale=5)  # * 3

        p[values.mask] = 0.0

        return p

    # pylint: disable=unused-variable
    continue_keep_speed_model = ModelExecutor(
        distance_to_pred_masked, movement_log_pdf, data
    )

    # create the movement models
    continue_keep_position_model = create_continue_keep_position_model(
        data, max_distance
    )
    split_movement_model = create_split_movement_model(data, max_distance)

    # pylint: disable=unused-variable
    split_rate_model = create_split_rate_model(data, subsampling)

    # create distance models for splits
    split_children_distance_model = create_split_children_distance_model(data)
    split_children_angle_model = create_split_children_angle_model(data)

    # create growth models
    continue_growth_model = create_growth_model(
        data, mean=1.008**subsampling, scale=0.6
    )
    split_growth_model = create_growth_model(data, mean=1.016**subsampling, scale=0.6)

    migration_models = [
        continue_keep_position_model,
        # continue_keep_speed_model,
        continue_growth_model,
    ]
    split_models = [
        split_movement_model,
        # split_rate_model,
        split_growth_model,
        # split_children_distance_model,
        # split_children_angle_model,
    ]  # , split_distance_ratio]

    assignment_generators = [
        SimpleNewAssGenerator([constant_new_model]),
        SimpleEndTrackAssGenerator([constant_end_model, end_pos_model]),
        # SimpleContinueGenerator(filters, [continue_growth_model, continue_rate_model, continue_movement_model]),
        SimpleContinueGenerator(filters, migration_models),
        # SimpleSplitGenerator(filters, partial(split_pair_filter, cdc=cdc), [split_rate_model_simple, split_growth_model, split_sum_growth_model, split_movement_model, split_dist_prop_model]) #, split_dist_prop_model])]# split_prop_model # split_growth_model, split_min_dist_model, split_rate_model_simple, split_prop_model
        SimpleSplitGenerator(
            filters + split_filters,
            partial(split_pair_filter, cdc=cdc, distance_threshold=max_distance, df=df),
            split_models,
        ),  # , split_dist_prop_model])]# split_prop_model # split_growth_model, split_min_dist_model, split_rate_model_simple, split_prop_model
    ]

    return assignment_generators
