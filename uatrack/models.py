"""Functionality for extracting information and scoring it according to probabilistic models"""

from __future__ import annotations

import numpy as np
from tensor_walks.impl_np import NP_Impl

backend = NP_Impl()


class ConstantModel:
    """
    model for constant probability
    """

    def __init__(self, const_log_value):
        self.const_log_value = const_log_value

    def __call__(self, tracking, source_index, target_index):
        return np.ones(source_index.shape[0]) * self.const_log_value


class ModelExecutor:
    """General model executor class
    It contains a quantity_computer and a (statistical) model.
    The quantity_computer extracts the information from suggested assignments (e.g. growth).
    The model computes a score for the extracted values."""

    def __init__(self, quantity_computer, model, df, name=None):
        self.quantity_computer = quantity_computer
        self.model = model
        self.df = df
        self.name = name

    def __call__(self, tracking, source_index, target_index):
        values = self.quantity_computer(tracking, source_index, target_index, self.df)

        return np.clip(self.model(values), -40, 0)

    def compute_quantities(self, tracking, source_index, target_index):
        values = self.quantity_computer(tracking, source_index, target_index, self.df)

        return values

    def __repr__(self):
        return f"ModelExecutor({self.name})"


# ---------------------------
# --- Quantity extractors ---
# ---------------------------


def area_growth_computer(tracking, source_index, target_index, df):
    """
    Computes the joint area growth between source and targets for all assignments
    """
    # pylint: disable=unused-argument

    assert source_index.shape[1] == 1 and len(source_index.shape) == 2
    assert len(target_index.shape) == 2

    # fetch area
    source_areas = df["area"][
        source_index.flatten()
    ]  # df['area'].to_numpy()[source_index.flatten()]
    target_areas = df["area"][target_index]  # .to_numpy()[target_index]

    # sum up the target areas
    target_joint_area = np.sum(target_areas, axis=-1)

    assert target_joint_area.shape == source_areas.shape

    # compute growth
    return np.divide(target_joint_area, source_areas)


def predict_movement(
    tracking,
    source_index,
    df,
    history: int,
    alpha=None,
    stop_at_split=True,
):
    """
    Predict the movement of a cell based on its past movements (in history timesteps). If no history is available assume no movement.

    Returns the distance between the prediction and the true positions

    """
    # fetch centroids
    all_centroids = df[
        "centroid"
    ]  # np.array(df['centroid'].to_list(), dtype=np.float32)

    # compute unique array
    unique_sources, inverse_indexing = np.unique(source_index, return_inverse=True)

    # perform walks from unique sources
    parents = backend.compute_walks_upward(
        tracking.parents, unique_sources, history, stop_at_split=stop_at_split
    )

    # get the centroids of the walks
    # walk_centroids = backend.compute_centroids(parents, all_centroids)

    # compute the average movement along the walks
    avg_movement = backend.compute_avg_property_along_walk(
        parents, all_centroids, exp_moving_average=alpha
    )

    # predict the next position of the sources
    pred_position = all_centroids[unique_sources] + avg_movement

    # rearrange predicted positions to assignment shape
    source_predictions = pred_position[inverse_indexing, :]

    # for all the sources without history we assume no movement
    pred_mask = np.all(source_predictions.mask, axis=1)
    source_predictions[pred_mask, :] = all_centroids[
        source_index[pred_mask].flatten(), :
    ]

    return source_predictions


def predict_property(
    tracking,
    source_index,
    df,
    history: int,
    alpha=None,
    stop_at_split=True,
    property_name="area",
):
    """
    Predicts the development of a single-cell property based on its past movements (in history timesteps). If no history is available assume no development.

    Returns the predicted property development
    """
    # fetch centroids
    all_properties = df[
        property_name
    ]  # np.array(df['centroid'].to_list(), dtype=np.float32)

    # auto compute property dimension
    prop_dim = 1
    if len(all_properties.shape) == 2:
        prop_dim = all_properties.shape[-1]

    # compute unique array
    unique_sources, inverse_indexing = np.unique(source_index, return_inverse=True)

    # perform walks from unique sources
    parents = backend.compute_walks_upward(
        tracking.parents, unique_sources, history, stop_at_split=stop_at_split
    )

    # get the centroids of the walks
    # walk_centroids = backend.compute_centroids(parents, all_centroids)

    # compute the average movement along the walks
    avg_development = backend.compute_avg_property_along_walk(
        parents, all_properties, exp_moving_average=alpha, distance_dims=prop_dim
    )

    # predict the next position of the sources
    pred_position = all_properties[unique_sources] + avg_development

    # rearrange predicted positions to assignment shape
    source_predictions = pred_position[inverse_indexing, ...]

    # for all the sources without history we assume no movement
    pred_mask = np.all(source_predictions.mask, axis=-1)
    if np.any(pred_mask):
        source_predictions[pred_mask] = all_properties[
            source_index[pred_mask].flatten()
        ]

    return source_predictions


def first_order_growth(
    tracking,
    source_index,
    target_index,
    df,
    history: int,
    alpha=None,
    stop_at_split=True,
):
    # extract all areas
    all_areas = df["area"]

    # assert target_index.shape[1] == 1

    # predict area development
    source_predictions = predict_property(
        tracking,
        source_index=source_index,
        df=df,
        history=history,
        stop_at_split=stop_at_split,
        property_name="area",
        alpha=alpha,
    )
    # source_predictions = np.repeat(source_predictions, target_index.shape[1], axis=0)

    target_areas = all_areas[target_index]  # .to_numpy()[target_index]

    # sum up the target areas
    target_joint_area = np.sum(target_areas, axis=-1)

    # print("Dimensions:")
    # print(source_predictions.shape, target_joint_area.shape)

    # compute area difference between sources and targets
    assignment_distances = source_predictions - target_joint_area

    # print(assignment_distances.shape)

    # return the distances
    return assignment_distances


def distance_to_pred_computer(
    tracking,
    source_index,
    target_index,
    df,
    history: int,
    alpha=None,
    stop_at_split=True,
):
    """
    Predict the movement of a cell based on its past movements (in history timesteps). If no history is available assume no movement.

    Returns the distance between the prediction and the true positions

    """
    # fetch centroids
    all_centroids = df["centroid"]

    # predict position by movement estimation
    source_predictions = predict_property(
        tracking,
        source_index=source_index,
        df=df,
        history=history,
        stop_at_split=stop_at_split,
        property_name="centroid",
        alpha=alpha,
    )

    # extract target positions (positions of cell detections in the next frame)
    target_positions = all_centroids[target_index]

    # compute distances between predicitions and target positions
    # we introduce a new dimensions because there can be multiple target positions (e.g. two for division into two children)
    assignment_distances = np.linalg.norm(
        source_predictions[:, None, :] - target_positions, axis=-1
    )

    # Sum multiple distances distances
    assignment_distances = np.sum(assignment_distances, axis=1)

    # return the distances
    return assignment_distances


def distance_to_pred_masked(tracking, source_index, target_index, df, num_steps=3):
    walks = backend.compute_walks_upward(
        tracking.parents,
        source_index.reshape(-1),
        num_steps=num_steps,
        stop_at_split=True,
    )

    all_positions = df["centroid"]

    avg_movement = backend.compute_avg_property_along_walk(walks, all_positions)

    pred_position = all_positions[source_index.reshape(-1)] + avg_movement

    target_position = all_positions[target_index.reshape(-1)]

    pred_distance = np.abs(np.linalg.norm(pred_position - target_position, axis=1))

    invalid = np.any(pred_position.mask)

    return np.ma.array(pred_distance, invalid)


def compute_dist_to_last_split(tracking, walk_matrix, no_split_behavior="mask"):
    """
    tracking: The current parent lookup table
    walk_matrix: NxM walk matrix for N sources and at max M steps
    no_split_behavior: if "mask" return masked array, otherwise invalid entries are filled with distance to root

    return distance array (N,)
    """
    # now we have to measure distance to last split
    numChildren = backend.compute_num_children(tracking)

    # the number of children for every parent
    data = np.ma.masked_array(
        numChildren[walk_matrix[:, 1:]], mask=~(walk_matrix[:, 1:] >= 0)
    )
    # np.argmax returns first occurence index of True, data > 1 means node did split
    all_dist = np.argmax(data > 1, axis=1)

    if no_split_behavior == "mask":
        all_dist = np.ma.masked_array(all_dist, mask=~(np.any(data > 1, axis=1)))
    else:
        invalid_entris = np.all(data <= 1, axis=1)

        # for all invalid entries compute the number of steps to root
        all_dist[invalid_entris] = np.sum((walk_matrix >= 0)[invalid_entris], axis=1)

    return all_dist


def split_dist_computer(
    tracking, source_index, target_index, df, max_time_consideration=100
):
    """
    Computes the number of timesteps to the last division event

    max_time_consideration: Maximum steps of time to reach into the past to find the parents

    returns the distance to the last split or -1 if the last split could not be found
    """
    # pylint: disable=unused-argument

    # fetch the unique source ideas
    unique_sources = np.unique(source_index)

    # initialize the distances to (-1): -1 means
    distances = np.ones(np.max(unique_sources) + 1, dtype=np.int32) * -1

    parents = backend.compute_walks_upward(
        tracking.parents, unique_sources, max_time_consideration
    )

    masked_distances = compute_dist_to_last_split(tracking.parents, parents)

    distances[unique_sources] = masked_distances.data

    distances[unique_sources[masked_distances.mask]] = -1

    return distances[source_index.flatten()]
