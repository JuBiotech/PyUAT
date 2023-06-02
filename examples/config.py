""" Configuration for the tracking model """

import logging
from functools import partial

import numpy as np
from scipy.stats import beta, binom, halfnorm
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
    source_index, target_index, cdc: ContourDistanceCache, distance_threshold=1.0
):
    """Make sure that distances for splitting are not too large"""
    # pylint: disable=unused-argument
    return cdc.distance(target_index[:, 0], target_index[:, 1]) < distance_threshold


def setup_assignment_generators(df, data, width):
    # TODO: Why these values?
    constant_new_model = ConstantModel(np.log(0.5**2))
    constant_end_model = ConstantModel(np.log(0.5**2))

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

    max_distance = 75

    def distance_to_previous(tracking, source_index, target_index, df):
        # pylint: disable=unused-argument
        all_centroids = data[
            "centroid"
        ]  # np.array(data['centroid'].to_list(), dtype=np.float32)

        sources = all_centroids[source_index]
        targets = all_centroids[target_index]

        return np.linalg.norm(sources - targets, axis=-1).squeeze()

    continue_keep_position_model = ModelExecutor(
        # partial(distance_to_pred_computer, alpha=1, history=0, stop_at_split=True),
        distance_to_previous,
        lambda values: beta.logsf(
            np.clip(values.squeeze() / max_distance, 0, 1), a=1, b=3
        ),
        # lambda values: expon.logsf(values, scale=5),
        data,
    )

    def movement_log_pdf(values):

        p = halfnorm.logsf(x=values, scale=5) * 3

        p[values.mask] = 0.0

        return p

    continue_keep_speed_model = ModelExecutor(
        distance_to_pred_masked, movement_log_pdf, data
    )

    def growth_probs(values, mean=1.06, scale=0.3):
        probs = halfnorm.logsf(np.abs(mean - values), scale=scale)
        return probs

    continue_growth_model = ModelExecutor(
        area_growth_computer,
        # growth_probs,
        partial(growth_probs, mean=1.008, scale=0.6),
        data,
    )

    split_growth_model = ModelExecutor(
        area_growth_computer,
        # partial(growth_probs, mean=1.2, scale=.6),
        partial(growth_probs, mean=1.016, scale=0.6),
        data,
    )

    split_movement_model = ModelExecutor(
        partial(distance_to_pred_computer, alpha=1, history=0, stop_at_split=True),
        lambda values: beta.logsf(
            np.clip(values.squeeze() / max_distance, 0, 1), a=1, b=3
        ).sum(axis=-1),
        # lambda values: expon.logsf(values, scale=5).sum(axis=-1),
        data,
    )

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

    migration_models = [
        continue_keep_position_model,
        continue_keep_speed_model,
        continue_growth_model,
    ]
    split_models = [
        split_movement_model,
        # split_rate_model,
        split_growth_model,
    ]  # , split_distance_ratio]

    assignment_generators = [
        SimpleNewAssGenerator([constant_new_model]),
        SimpleEndTrackAssGenerator([constant_end_model, end_pos_model]),
        # SimpleContinueGenerator(filters, [continue_growth_model, continue_rate_model, continue_movement_model]),
        SimpleContinueGenerator(filters, migration_models),
        # SimpleSplitGenerator(filters, partial(split_pair_filter, cdc=cdc), [split_rate_model_simple, split_growth_model, split_sum_growth_model, split_movement_model, split_dist_prop_model]) #, split_dist_prop_model])]# split_prop_model # split_growth_model, split_min_dist_model, split_rate_model_simple, split_prop_model
        SimpleSplitGenerator(
            filters,
            partial(split_pair_filter, cdc=cdc, distance_threshold=max_distance),
            split_models,
        ),  # , split_dist_prop_model])]# split_prop_model # split_growth_model, split_min_dist_model, split_rate_model_simple, split_prop_model
    ]

    return assignment_generators
