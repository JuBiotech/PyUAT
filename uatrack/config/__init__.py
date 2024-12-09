""" Configuration for the tracking model """
from __future__ import annotations

import logging
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import binom, halfnorm, norm
from tensor_walks.impl_np import NP_Impl

from uatrack.assignment import (
    SimpleAssignmentGenerator,
    SimpleContinueGenerator,
    SimpleEndTrackAssGenerator,
    SimpleNewAssGenerator,
    SimpleSplitGenerator,
)
from uatrack.models import (
    ConstantModel,
    ModelExecutor,
    area_growth_computer,
    distance_to_pred_computer,
    first_order_growth,
    predict_property,
    split_dist_computer,
)
from uatrack.utils import ContourDistanceCache, NearestNeighborCache

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
        "Children split distance",
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


def prob_cont_angles(angles, scale):

    # compute the probability of more extreme values
    prob = halfnorm.logsf(
        angles, scale=scale
    )  # log_sum(norm.logcdf(loc - diff, loc=loc, scale=scale), norm.logsf(loc + diff, loc=loc, scale=scale))

    # do not use nan values
    prob[np.isnan(prob)] = -30

    return prob


def create_continue_angle_model(data, prob):
    # pylint: disable=unused-argument
    def continue_angle_extractor(
        tracking,
        source_index,
        target_index,
        df,
    ):
        major_axis = df["major_axis"]

        majA = major_axis[source_index.flatten()]
        majB = major_axis[target_index.flatten()]

        vec_norm = np.linalg.norm(majA, axis=-1) / np.linalg.norm(majB, axis=-1)

        raw_values = np.stack(
            [
                [np.dot(a, b) for a, b in zip(majA, majB)] / vec_norm,
                [np.dot(-a, b) for a, b in zip(majA, majB)] / vec_norm,
            ],
            axis=-1,
        )

        raw_values = np.clip(raw_values, -1.0, 1.0)

        angles = np.min(np.arccos(raw_values) * 180 / np.pi, axis=-1)

        return angles

    continue_angle_model = ModelExecutor(continue_angle_extractor, prob, data)

    return continue_angle_model


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


def create_first_order_growth_model(data, scale, history=1):
    return ModelExecutor(
        partial(first_order_growth, history=history),
        lambda x: log_sum(
            norm.logcdf(x, loc=0, scale=scale), norm.logsf(x, loc=0, scale=scale)
        ),
        data,
        "First order growth",
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
        # partial(distance_to_pred_computer, alpha=1, history=0, stop_at_split=True),
        distance_to_previous,
        prob,
        # lambda values: beta.logsf(
        #    np.clip(values.squeeze() / max_distance, 0, 1), a=1, b=3
        # ),
        # lambda values: expon.logsf(values, scale=5),
        data,
    )


def dis_app_prob_func(x_coordinates, max_overshoot, min_dist, x_size: int):
    total_auc = 0.5 * (min_dist + max_overshoot) ** 2
    area_func = lambda x: 0.5 * (x + max_overshoot) ** 2
    prob = np.log(
        1
        - area_func(
            np.clip(
                np.min(np.stack([x_coordinates, x_size - x_coordinates]), axis=0),
                -max_overshoot,
                min_dist,
            )
        )
        / total_auc
    )

    prob[np.isnan(prob)] = -30

    return prob


# pylint: disable=unused-argument
def source_position_extractor(
    tracking,
    source_index,
    target_index,
    df,
):
    # extract the x-position of all sources
    return df["centroid"][source_index.flatten()][:, 1]


def create_disappear_model(data, width: int):
    return ModelExecutor(
        source_position_extractor,
        partial(dis_app_prob_func, max_overshoot=20, min_dist=20, x_size=width),
        data,
    )


def create_continue_temp_position_model(data, prob, history=3):
    return ModelExecutor(
        partial(
            distance_to_pred_computer, alpha=None, history=history, stop_at_split=True
        ),
        prob,
        data,
        "First order position model",
    )


def create_split_movement_model(data, prob):
    return ModelExecutor(
        distance_to_previous,
        prob,
        data,
    )


def migration_prob(data: dict[str, np.ndarray], subsampling: int):
    """Model for cell migration probability (tailored to specific sequence).

    Args:
        data (dict[str, np.ndarray]): single-cell data
        subsampling (int): subsampling factor

    Returns:
        _type_: _description_
    """

    # 1. data < 0: no division has been observed yet -> cell could divide at any moment (p=0.5)
    # 2. 0 <= data < 40: cell is quite young (p=0.9)
    # 3. 40 <= data < 81: cell is in division age (p=0.1)
    # 4. 81 <= data: cell is over typical division age. We do not know what's going on (p=0.5)
    print(f"data dim: {data.shape}")
    # expected division timeframe between [min_time, max_time]. Divided by subsampling factor
    min_time = 40 / subsampling
    max_time = 80 / subsampling

    # probability of migration is high before division time window. During division time window it is low. After that it is unknown (0.5)
    return (
        (data < 0) * 0.5
        + ((0 <= data) & (data < min_time)) * 0.9
        + ((min_time <= data) & (data <= max_time)) * 0.1
        + (max_time < data) * 0.5
    )


def create_age_migration_model(
    data: dict[str, np.ndarray], subsampling: int
) -> ModelExecutor:
    """Creates migration probability model based on cell age

    Args:
        data (dict[str, np.ndarray]): single-cell data
        subsampling (int): subsampling factor

    Returns:
        ModelExecutor: scoring model
    """
    return ModelExecutor(
        split_dist_computer, lambda x: np.log(migration_prob(x, subsampling)), data
    )


def create_age_division_model(
    data: dict[str, np.ndarray], subsampling: int
) -> ModelExecutor:
    """Create age division model.
       This probability is derived from the probability not to migrate

    Args:
        data (dict[str, np.ndarray]): single-cell data
        subsampling (int): subsamping factor

    Returns:
        ModelExecutor: scoring model executor
    """
    return ModelExecutor(
        split_dist_computer, lambda x: np.log(1 - migration_prob(x, subsampling)), data
    )


def create_end_pos_model(
    data: dict[str, np.ndarray], width: int, subsampling: int, history=1
) -> ModelExecutor:
    """Create model for predicting cell leaving the field of view. This lowers cell disappearance probabilites at the center of the microscopy image (and far from the vertical edges)

    Args:
        data (dict[str, np.ndarray]): single-cell data
        width (int): width of the image
        subsampling (int): subsampling factor
        history (int, optional): number of historic cell movements to use for position prediction. Defaults to 1.
    """

    def position(tracking, source_index, target_index, df):
        # pylint: disable=unused-argument

        # predict cell movement
        source_predictions = predict_property(
            tracking,
            source_index=source_index,
            df=df,
            history=history,
            stop_at_split=True,
            property_name="centroid",
        )

        return source_predictions

    def pos_log_pdf(values):
        values = values[:, 0]
        min_dist = np.minimum(values, np.abs(width - values))

        # all cells have a standard disappear probability
        p = np.zeros_like(values) + np.log(0.25)
        # all cells far from the border have a lower disappear prob
        p[min_dist > np.clip(10 * subsampling, 0, 100)] = -4  # -6 #np.log(1e-6)

        return p

    return ModelExecutor(position, pos_log_pdf, data)


class SimpleCDC:
    """class to compute contour distances (approximated by center position in favor of speed)"""

    def __init__(self, positions):
        self.positions = positions

    def distance(self, indA, indB):
        return np.linalg.norm(self.positions[indA] - self.positions[indB], axis=1)


def create_constant_disappear_model(app_prob: float):
    return ConstantModel(np.log(app_prob)), ConstantModel(np.log(app_prob))


def standard_models(
    data,
    subsampling,
    use_long_hist=0,
    app_prob=0.25,
    mig_growth_scale=0.05,
    mig_movement_scale=20,
    div_growth_scale=None,
    div_movement_scale=20,
    mig_growth_mean=1.008,
    div_growth_mean=1.016,
):

    if div_growth_scale is None:
        div_growth_scale = 2 * mig_growth_scale
    if div_movement_scale is None:
        div_movement_scale = 2 * mig_movement_scale

    # dis (-appear) models
    constant_new_model, constant_end_model = create_constant_disappear_model(app_prob)

    # Movement models
    continue_movement_model = create_continue_temp_position_model(
        data,
        prob=lambda val: halfnorm.logsf(
            val.flatten(), scale=mig_movement_scale * subsampling
        ),
        history=use_long_hist,
    )
    # continue_movement_model = create_continue_keep_position_model(
    #    data, prob=lambda val: halfnorm.logsf(val, scale=mig_movement_scale*subsampling)
    # )
    split_movement_model = create_split_movement_model(
        data,
        prob=lambda val: halfnorm.logsf(val, scale=div_movement_scale * subsampling),
    )

    # growth models:
    continue_growth_model = create_growth_model(
        data, mean=mig_growth_mean**subsampling, scale=mig_growth_scale * subsampling
    )
    split_growth_model = create_growth_model(
        data, mean=div_growth_mean**subsampling, scale=div_growth_scale * subsampling
    )

    return (
        [constant_new_model],
        [constant_end_model],
        [continue_movement_model, continue_growth_model],
        [split_movement_model, split_growth_model],
    )


def create_filters(
    data: dict[str, np.ndarray], df: pd.DataFrame, k=15
) -> tuple[list, SimpleCDC]:
    """Creates a nearest neighbor filter and a distance computer

    Args:
        data (dict[str, np.ndarray]): single-cell data
        df (pd.DataFrame): single-cell data (in table format)
        k (int, optional): Number of nearest neighbors to consider. Defaults to 15.

    Returns:
        tuple[list, SimpleCDC]: tuple of nearest neighbor filter and distance computer
    """

    positions = data["centroid"]
    cdc = SimpleCDC(positions)

    logging.info("Compute nearest neighbor cache")
    nnc = NearestNeighborCache(df)

    filters = [lambda s, t: nnc.kNearestNeigborsMatrixMask(k, s, t)]

    return filters, cdc


def use_disappear_model(
    data: dict[str, np.ndarray], subsampling: int, width: int, **kwargs
) -> tuple[
    list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]
]:
    """Adds disappear models to the first-order models

    Args:
        data (dict[str, np.ndarray]): single-cell data (numpy arrays)
        subsampling (int): subsampling factor. Used to linearly scale the scale value of statistical distributions
        width (int): width of the microscopy image (in pixels)

    Returns:
        tuple[list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]]: list of scoring models for appearance, disappearance, migration and division models
    """

    # make standard models
    (
        constant_new_models,
        _,
        migration_models,
        split_models,
    ) = use_first_order_model(data=data, subsampling=subsampling, **kwargs)

    end_pos_model = create_end_pos_model(data, width, subsampling)

    return constant_new_models, [end_pos_model], migration_models, split_models


def add_split_distance_model(
    data: dict[str, np.ndarray], subsampling: int, **kwargs
) -> tuple[
    list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]
]:
    """Adds division distance model to the first-order models

    Args:
        data (dict[str, np.ndarray]): single-cell data (numpy arrays)
        subsampling (int): subsampling factor. Used to linearly scale the scale value of statistical distributions

    Returns:
        tuple[list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]]: list of scoring models for appearance, disappearance, migration and division models
    """

    # make first-order models
    (
        constant_new_models,
        constant_end_models,
        migration_models,
        split_models,
    ) = use_first_order_model(data=data, subsampling=subsampling, **kwargs)

    # add children distance model
    split_children_distance_model = create_split_children_distance_model(
        data, prob=lambda vs: halfnorm.logsf(vs, loc=0, scale=3 * subsampling)
    )
    split_models += [split_children_distance_model]

    return constant_new_models, constant_end_models, migration_models, split_models


def add_angle_models(
    data: dict[str, np.ndarray], subsampling: int, **kwargs
) -> tuple[
    list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]
]:
    """Adds migration and division orientation models to the first-order models

    Args:
        data (dict[str, np.ndarray]): single-cell data (numpy arrays)
        subsampling (int): subsampling factor. Used to linearly scale the scale value of statistical distributions

    Returns:
        tuple[list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]]: list of scoring models for appearance, disappearance, migration and division models
    """

    # make first-order models
    (
        constant_new_models,
        constant_end_models,
        migration_models,
        split_models,
    ) = use_first_order_model(data=data, subsampling=subsampling, **kwargs)

    continue_angle_model = create_continue_angle_model(
        data, prob=partial(prob_cont_angles, scale=20 * subsampling)
    )
    split_children_angle_model = create_split_children_angle_model(
        data, prob=partial(prob_angles, loc=135, scale=20 * subsampling)
    )

    # add models
    migration_models += [continue_angle_model]
    split_models += [split_children_angle_model]

    return constant_new_models, constant_end_models, migration_models, split_models
    # return make_assignmet_generators(df, data, constant_new_model, constant_end_model, migration_models, split_models)


def add_age_models(
    data: dict[str, np.ndarray], subsampling: int, **kwargs
) -> tuple[
    list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]
]:
    """Adds the age model to the first-order models

    Args:
        data (dict[str, np.ndarray]): single-cell data (numpy arrays)
        subsampling (int): subsampling factor. Used to linearly scale the scale value of statistical distributions

    Returns:
        tuple[list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]]: list of scoring models for appearance, disappearance, migration and division models
    """

    # make first-order models
    (
        constant_new_models,
        constant_end_models,
        migration_models,
        split_models,
    ) = use_first_order_model(data=data, subsampling=subsampling, **kwargs)

    # create age models
    migration_age_model, split_age_model = (
        create_age_migration_model(data, subsampling),
        create_age_division_model(data, subsampling),
    )

    # add models
    migration_models += [migration_age_model]
    split_models += [split_age_model]

    return constant_new_models, constant_end_models, migration_models, split_models


def use_nearest_neighbor(
    data: dict[str, np.ndarray],
    subsampling: int,
    mig_growth_scale=0.05,
    mig_movement_scale=20,
    div_growth_scale: float = None,
    div_movement_scale=20.0,
    **kwargs,
) -> tuple[
    list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]
]:
    """Nearest neighbor tracking configuration assumes that cells do not move or grow and builds statistical distributions to score cell development based on this assumption.

    Args:
        data (dict[str, np.ndarray]): single-cell data (numpy arrays)
        subsampling (int): subsampling factor. Used to linearly scale the scale value of statistical distributions
        mig_growth_scale (float, optional): scale parameter for migration growth. Defaults to 0.05.
        mig_movement_scale (_type_, optional): scale parameter for migration movement (in pixels). Defaults to 20..
        div_growth_scale (float, optional): scale parameter for division growth. Defaults to None.
        div_movement_scale (_type_, optional): scale parameter for division movement. Defaults to 20.

    Returns:
        tuple[list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]]: list of scoring models for appearance, disappearance, migration and division models
    """

    if div_growth_scale is None:
        div_growth_scale = 2 * mig_growth_scale
    if div_movement_scale is None:
        div_movement_scale = 2 * mig_movement_scale

    # make standard models
    new_models, end_models, _, _ = standard_models(
        data=data, subsampling=subsampling, **kwargs
    )

    # Movement models: with history=0 --> Makes them expect no movement -> look for cell association at the same position
    continue_movement_model = create_continue_temp_position_model(
        data,
        prob=lambda val: halfnorm.logsf(
            val.flatten(), scale=mig_movement_scale * subsampling
        ),
        history=0,
    )
    split_movement_model = create_continue_temp_position_model(
        data,
        prob=lambda val: halfnorm.logsf(val, scale=div_movement_scale * subsampling),
        history=0,
    )

    # growth models: Assume no growth (simplification) --> look for cells with same size
    continue_growth_model = create_growth_model(
        data, mean=1, scale=mig_growth_scale * subsampling
    )
    split_growth_model = create_growth_model(
        data, mean=1, scale=div_growth_scale * subsampling
    )

    migration_models = [continue_movement_model, continue_growth_model]
    split_models = [split_movement_model, split_growth_model]

    return new_models, end_models, migration_models, split_models


def use_first_order_model(
    data: dict[str, np.ndarray],
    subsampling: int,
    use_long_hist=1,
    fo_mov_scale=5,
    fo_growth_scale=10,
    **kwargs,
) -> tuple[
    list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]
]:
    """Create first order-tracking configuration models.
       Idea: movement and growth are predicted based on past cell movement and growth.

    Args:
        data (dict[str, np.ndarray]): single-cell data (numpy arrays)
        subsampling (int): subsampling factor. Used to linearly scale the scale value of statistical distributions
        use_long_hist (int, optional): the track length used to predict single-cell properties. Defaults to 1.
        fo_mov_scale (float, optional): scale parameter for half-normal movement distribution. Defaults to 5.
        fo_growth_scale (float, optional): scale parameter for half-normal growth distribution. Defaults to 10.

    Returns:
        tuple[list[ModelExecutor], list[ModelExecutor], list[ModelExecutor], list[ModelExecutor]]: list of scoring models for appearance, disappearance, migration and division models
    """

    # make standard models
    constant_new_models, constant_end_models, _, _ = standard_models(
        data=data, subsampling=subsampling, **kwargs
    )

    # First order movement models
    continue_movement_model = create_continue_temp_position_model(
        data,
        lambda val: halfnorm.logsf(val, scale=20 + fo_mov_scale * subsampling),
        history=use_long_hist,
    )  # first order movement model
    continue_growth_model = create_first_order_growth_model(
        data, 50 + fo_growth_scale * subsampling, history=use_long_hist
    )  # None # first order growth model

    # Division models (only look at children distance & growth) -> divisions are restricted by combinations
    split_movement_model = create_continue_temp_position_model(
        data,
        lambda val: halfnorm.logsf(
            val, loc=0, scale=20 + 2 * fo_mov_scale * subsampling
        ),
        history=use_long_hist,
    )  # create_continue_temp_position_model(data, lambda val: halfnorm.logsf(val, scale=10*subsampling), history=1) #create_split_children_distance_model(data, prob=lambda vs: halfnorm.logsf(vs, loc=0, scale=3*subsampling)) #None # first order movement model
    split_growth_model = create_first_order_growth_model(
        data, 50 + 2 * fo_growth_scale * subsampling, history=use_long_hist
    )  # first order growth model (e.g. $(sum child area)=$(historical growth rate) * 2)

    # collect models
    migration_models = [
        continue_movement_model,
        continue_growth_model,
    ]
    split_models = [
        split_movement_model,
        split_growth_model,
    ]

    return constant_new_models, constant_end_models, migration_models, split_models


def add_growth_model(
    data,
    subsampling,
    mig_growth_scale=0.05,
    div_growth_scale=None,
    mig_growth_mean=1.008,
    div_growth_mean=1.016,
    **kwargs,
):

    new_models, end_models, migration_models, split_models = use_first_order_model(
        data=data, subsampling=subsampling, **kwargs
    )

    if div_growth_scale is None:
        div_growth_scale = 2 * mig_growth_scale

    # growth models:
    continue_growth_model = create_growth_model(
        data, mean=mig_growth_mean**subsampling, scale=mig_growth_scale * subsampling
    )
    split_growth_model = create_growth_model(
        data, mean=div_growth_mean**subsampling, scale=div_growth_scale * subsampling
    )

    # replace first-order growth models by bioligacally motivated models
    migration_models[1] = continue_growth_model
    split_models[1] = split_growth_model

    return new_models, end_models, migration_models, split_models


def make_assignmet_generators(
    df: pd.DataFrame,
    data: dict[str, np.ndarray],
    constant_new_models: list[ModelExecutor],
    constant_end_models: list[ModelExecutor],
    migration_models: list[ModelExecutor],
    split_models: list[ModelExecutor],
    max_distance=200,
) -> list[SimpleAssignmentGenerator]:
    """Creates a list of assignment generators based on the scoring models and basic nearest neighbor filtering

    Args:
        df (pd.DataFrame): single-cell data (table)
        data (dict[str, np.ndarray]): single-cell data (numpy arrays)
        constant_new_models (list[ModelExecutor]): statistical models for cell appearance
        constant_end_models (list[ModelExecutor]): statistical models for cell disappearance
        migration_models (list[ModelExecutor]): statistical models for cell migration
        split_models (list[ModelExecutor]): statistical models for cell division
        max_distance (int, optional): maximum distance of cells (in pixel) to be considered in assignments. Defaults to 200.

    Returns:
        list[SimpleAssignmentGenerator]: assignment generators
    """
    filters, cdc = create_filters(data, df)

    # create assignment generators
    return [
        SimpleNewAssGenerator(constant_new_models),
        SimpleEndTrackAssGenerator(constant_end_models),
        SimpleContinueGenerator(filters, migration_models),
        SimpleSplitGenerator(
            filters,
            partial(split_pair_filter, cdc=cdc, distance_threshold=max_distance, df=df),
            split_models,
        ),
    ]


def setup_assignment_generators(
    df: pd.DataFrame, subsampling_factor: int, model: str = "NN"
) -> list[SimpleAssignmentGenerator]:
    """Create the assignment generators and cell models for scoring tracking assignments

    Args:
        df (pd.DataFrame): data frame with single-cell data from segementation
        subsampling_factor (int): subsampling factor
        model (str, optional): the model configuration for the tracking. Defaults to "NN".

    Returns:
        list[SimpleAssignmentGenerator]: List of assignment generators equipped with cell models
    """

    # arrange single-cell information into numpy arrays (greatly increases the speed, as data can be immediately indexed)
    data = {
        "area": np.array(df["area"].to_list(), dtype=np.float32),
        "centroid": np.array(df["centroid"].to_list(), dtype=np.float32),
        "major_extents": np.array(df["major_extents"].to_list(), dtype=np.float32),
        "major_axis": np.array(df["major_axis"].to_list(), dtype=np.float32),
    }

    subsampling = subsampling_factor

    # create biologically motivated models
    if model == "NN":
        (
            constant_new_models,
            constant_end_models,
            migration_models,
            split_models,
        ) = use_nearest_neighbor(data=data, subsampling=subsampling_factor)
    elif model == "FO":
        (
            constant_new_models,
            constant_end_models,
            migration_models,
            split_models,
        ) = use_first_order_model(data=data, subsampling=subsampling_factor)
    elif model == "FO+G":
        (
            constant_new_models,
            constant_end_models,
            migration_models,
            split_models,
        ) = add_growth_model(
            data=data, subsampling=subsampling_factor
        )  # add_angle_models(data=data, subsampling=subsampling_factor) #use_first_order_model(data=data, subsampling=subsampling_factor) #use_nearest_neighbor(data=data, subsampling=subsampling_factor) #
    elif model == "FO+O":
        (
            constant_new_models,
            constant_end_models,
            migration_models,
            split_models,
        ) = add_angle_models(data=data, subsampling=subsampling_factor)
    elif model == "FO+DD":
        # make standard models
        (
            constant_new_models,
            constant_end_models,
            migration_models,
            split_models,
        ) = use_first_order_model(data=data, subsampling=subsampling)

        # add children distance model
        split_children_distance_model = create_split_children_distance_model(
            data, prob=lambda vs: halfnorm.logsf(vs, loc=0, scale=3 * subsampling)
        )
        split_models += [split_children_distance_model]

    elif model == "FO+G+O+DD":

        (
            constant_new_models,
            constant_end_models,
            migration_models,
            split_models,
        ) = add_growth_model(data=data, subsampling=subsampling_factor)

        # add children distance model
        split_children_distance_model = create_split_children_distance_model(
            data, prob=lambda vs: halfnorm.logsf(vs, loc=0, scale=3 * subsampling)
        )
        split_models += [split_children_distance_model]

        # add orientation models
        continue_angle_model = create_continue_angle_model(
            data, prob=partial(prob_cont_angles, scale=20 * subsampling)
        )
        split_children_angle_model = create_split_children_angle_model(
            data, prob=partial(prob_angles, loc=135, scale=20 * subsampling)
        )

        migration_models += [continue_angle_model]
        split_models += [split_children_angle_model]

    # create the assignment candidate generators
    assignment_generators = make_assignmet_generators(
        df=df,
        data=data,
        constant_new_models=constant_new_models,
        constant_end_models=constant_end_models,
        migration_models=migration_models,
        split_models=split_models,
    )

    return assignment_generators
