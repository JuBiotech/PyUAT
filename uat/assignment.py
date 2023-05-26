""" Module for assignment generation """

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


def filterIndexLists(source_index, target_index, filters):
    """
    That is for candidate filtering

    source_index
    target_index: index array SxT, where S is the number of possible sources and T is the number of possible targets
    """
    num_sources = len(source_index)
    num_targets = target_index.shape[1]

    # todo apply nearest neighbor filter here
    if len(filters) == 0:
        combined_mask = np.ones((num_sources, num_targets), dtype=np.bool)
        # combined_mask[:,0] = False
    else:
        stacked_masks = np.array([f(source_index, target_index) for f in filters])
        # print('stacke mask', stacked_masks.shape)
        # print(stacked_masks.dtype)
        combined_mask = np.all(stacked_masks, axis=0)
        # print('comb mask', combined_mask.shape)

    # source_index_new = np.repeat(source_index, (num_targets,)).reshape(num_sources * num_targets, -1)[combined_mask.flatten()].reshape((-1, 1))
    # print(target_index)
    # print(combined_mask)
    # print(combined_mask.dtype)
    # print('target_index', target_index.shape)
    mask_array = np.ma.masked_array(target_index, mask=~combined_mask)
    # print(mask_array)
    target_index_new = mask_array.compressed().reshape((num_sources, -1))

    # print(target_index_new.shape)
    # print(target_index_new)

    return source_index, target_index_new


class SimpleAssignmentGenerator:
    """
    Base class for assignment generators
    """

    def __init__(self, models: list):
        """Create an assignment generator

        Args:
            models (list): List of models to score the assignments
        """
        self.models = models

    def generate(self, tracking, sources, targets):
        raise NotImplementedError()

    def compute_scores(
        self,
        tracking: np.ndarray[Any, np.uint32],
        source_index: np.ndarray[(Any, 1), np.uint32],
        target_index: np.ndarray[(Any, Any), np.uint32],
    ) -> tuple[np.float32, np.ndarray[(Any, Any), np.float32]]:
        """Compute the scores for all assignments

        Args:
            tracking (np.ndarray[Any, np.uint32]): index based tracking representation
            source_index (np.ndarray[(Any, 1), np.uint32]) : sources for the assignemnts
            target_index (np.ndarray[(Any, Any), np.uint32]): targets for the assignment

        Raises:
            ValueError: we need models for scoring

        Returns:
            _type_: sum of scores and a list of individual scores
        """

        if len(self.models) <= 0:
            raise ValueError("You need at least one model for scoring assignments")

        scores = np.array(
            [m(tracking, source_index, target_index) for m in self.models]
        )
        assert len(scores.shape) == 2
        return np.sum(scores, axis=0), scores.T


class SimpleNewAssGenerator(SimpleAssignmentGenerator):
    """
    Generates new detection assignments
    """

    def generate(self, tracking, sources, targets) -> tuple:
        # for new detections the source is -1
        source_index = np.ones((len(targets), 1), dtype=np.int32) * -1

        # the target indices are simple the targets but with 2 dimensions
        target_index = np.zeros((len(targets), 1), dtype=np.int32)
        target_index[:, 0] = targets

        summed_scores, individual_scores = self.compute_scores(
            tracking, source_index, target_index
        )

        return (
            source_index,
            target_index,
            summed_scores,
            individual_scores,
        )  # np.ones((source_index.shape[0])) * -7


class SimpleEndTrackAssGenerator(SimpleAssignmentGenerator):
    """
    Generates new detection assignments
    """

    def generate(self, tracking, sources, targets) -> tuple:
        # for new detections the source is -1
        source_index = np.zeros((len(sources), 1), dtype=np.int32)
        source_index[:, 0] = sources.index

        # the target indices are simple the targets but with 2 dimensions
        target_index = np.zeros((len(sources), 0), dtype=np.int32)
        # target_index[:,0] = targets.index

        summed_scores, individual_scores = self.compute_scores(
            tracking, source_index, target_index
        )

        return (
            source_index,
            target_index,
            summed_scores,
            individual_scores,
        )  # np.ones((source_index.shape[0])) * -7


class SimpleContinueGenerator(SimpleAssignmentGenerator):
    """Generator for continue assignments"""

    def __init__(self, candidate_filters, models):
        """
        candidate_filters: filter possible target candidates w.r.t to a source
        models: models for scoring the assignments
        """
        super().__init__(models)
        self.candidate_filters = candidate_filters

    def generate(self, tracking, sources, targets):
        if len(sources) == 0:
            return (
                np.zeros((0, 0), dtype=np.uint32),
                np.zeros((0, 0), dtype=np.uint32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        source_index = sources.index.to_numpy()
        target_index = np.tile(targets.index.to_numpy(), (len(sources), 1))

        source_index, target_index = filterIndexLists(
            source_index, target_index, self.candidate_filters
        )

        source_index = np.repeat(source_index, target_index.shape[1]).reshape((-1, 1))
        target_index = target_index.reshape(-1, 1)

        assert source_index.shape == target_index.shape
        assert source_index.shape[1] == 1

        # sum up the logarithms (product in prob space)
        summed_scores, individual_scores = self.compute_scores(
            tracking, source_index, target_index
        )  # np.sum([m(tracking, source_index, target_index) for m in self.models], axis=0)

        assert summed_scores.shape[0] == source_index.shape[0]

        return source_index, target_index, summed_scores, individual_scores


class SimpleSplitGenerator(SimpleAssignmentGenerator):
    """Generator for division assignments"""

    def __init__(self, candidate_filters, pair_filter, models):
        super().__init__(models)
        self.candidate_filters = candidate_filters
        self.pair_filter = pair_filter

    def generate(self, tracking, sources, targets):
        if len(sources) == 0:
            return (
                np.zeros((0, 0), dtype=np.uint32),
                np.zeros((0, 0), dtype=np.uint32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        source_index = sources.index.to_numpy()
        # create target matrix
        target_index = np.tile(np.array(targets.index), (len(sources), 1))

        # apply filters for target indices
        source_index, target_index = filterIndexLists(
            source_index, target_index, self.candidate_filters
        )

        combinations = np.array(
            [list(itertools.combinations(targets, 2)) for targets in target_index]
        )

        source_index = np.repeat(
            sources.index.to_numpy(), (combinations.shape[1],)
        ).reshape((-1, 1))
        target_index = combinations.reshape((-1, 2))

        # filter lists to only appropriate assignments
        mask = self.pair_filter(source_index, target_index)

        source_index = source_index[mask]
        target_index = target_index[mask]

        # print(source_index, target_index)
        assert source_index.shape[0] == target_index.shape[0]
        assert source_index.shape[1] == 1
        assert target_index.shape[1] == 2

        # sum up the logarithms (product in prob space)
        summed_scores, individual_scores = self.compute_scores(
            tracking, source_index, target_index
        )  # np.sum([m(tracking, source_index, target_index) for m in self.models], axis=0)

        assert source_index.shape[0] == summed_scores.shape[0]
        assert len(summed_scores.shape) == 1

        return source_index, target_index, summed_scores, individual_scores
