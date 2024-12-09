""" Module for assignment generation """

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np


def filter_targets(source_index, target_index, filters):
    """Filter targets

    Args:
        source_index (_type_): index array for the sources
        target_index (_type_): index array of SxT (source, filtered candidates)
        filters (_type_): filter functions returning boolean arrays for (source_index, target_index)

    Returns:
        tuple: [source_index, filtered target_index]
    """

    num_sources = len(source_index)
    num_targets = target_index.shape[1]

    if len(filters) == 0:
        # no filters at all
        combined_mask = np.ones((num_sources, num_targets), dtype=bool)
    else:
        # apply all filters to the source, index combinations
        stacked_masks = np.array([f(source_index, target_index) for f in filters])
        # only when all filters aggree use the
        combined_mask = np.all(stacked_masks, axis=0)

    # pylint: disable=singleton-comparison
    if np.all(combined_mask == False):
        logging.warning("Filtered all away!")

    # just keep the valid (source, target) combinations
    mask_array = np.ma.masked_array(target_index, mask=~combined_mask)
    target_index_new = mask_array.compressed().reshape((num_sources, -1))

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
        )


class SimpleEndTrackAssGenerator(SimpleAssignmentGenerator):
    """
    Generates new detection assignments
    """

    def generate(self, tracking, sources, targets) -> tuple:
        # for dissappearing cells copy the sources
        source_index = np.zeros((len(sources), 1), dtype=np.int32)
        source_index[:, 0] = sources

        # the target index has dim=0 (no connections)
        target_index = np.zeros((len(sources), 0), dtype=np.int32)

        # compute the scores
        summed_scores, individual_scores = self.compute_scores(
            tracking, source_index, target_index
        )

        # return all variables
        return (
            source_index,
            target_index,
            summed_scores,
            individual_scores,
        )


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
            # the case for no sources (e.g. start of the tracking): no continue assignments
            return (
                np.zeros((0, 0), dtype=np.uint32),
                np.zeros((0, 0), dtype=np.uint32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        # setup index lists
        source_index = sources
        # targets are TxSx1
        target_index = np.tile(targets, (len(sources), 1))

        source_index, target_index = filter_targets(
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

        source_index = sources
        # create target matrix
        target_index = np.tile(np.array(targets), (len(sources), 1))

        # apply filters for target indices
        source_index, target_index = filter_targets(
            source_index, target_index, self.candidate_filters
        )

        combinations = np.array(
            [list(itertools.combinations(targets, 2)) for targets in target_index]
        )

        source_index = np.repeat(sources, (combinations.shape[1],)).reshape((-1, 1))
        target_index = combinations.reshape((-1, 2))

        # filter lists to only appropriate assignments
        mask = self.pair_filter(source_index, target_index)

        source_index = source_index[mask]
        target_index = target_index[mask]

        # check whether we still have opportunities
        if len(source_index) == 0 or len(target_index) == 0:
            # we have no split opportunities
            print("no splits available!")
            return (
                np.zeros((0, 0), dtype=np.uint32),
                np.zeros((0, 0), dtype=np.uint32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

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
