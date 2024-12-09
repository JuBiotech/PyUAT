""" This module contains the core functionality """

from __future__ import annotations

import copy
import json
import logging
import multiprocessing
import time
from functools import reduce
from pathlib import Path

import numpy as np
import numpy.typing as npt
import ray
from networkx import DiGraph
from pandas import DataFrame
from tqdm.auto import tqdm

from uatrack.sampling import sample_from_probabilities
from uatrack.solve.gurobi import SimpleGurobiSolver
from uatrack.solve.mip import SimpleMIPSolver

from .sampling import Probability
from .utils import ContourDistanceCache


class SimpleTracking:
    """Contains a tracking solution in simple array formats"""

    def __init__(self, tracking, parent_id=-1, id=-1, active_assignments=None):
        self.id = id
        self.parent_id = parent_id
        self.parents = tracking

        if active_assignments is None:
            active_assignments = []

        self.active_assignments = active_assignments

    def compute_trackends(self, lastFrameIndices):
        """
        computes the current active trackends

        lastFrameIndices: all indices the are candidates for being track ends
        """

        # trackends are those that have a parent, or roots (-1)
        return lastFrameIndices[self.parents[lastFrameIndices] >= -1]

    def expand(self, solution, assignments):
        """
        Expend the current tracking with a new solution

        solution: solution structure (mask, score)
        assignments: corresponding assignment structure
        """
        mask, _ = solution
        assignments = zip(*assignments)

        new_parents = np.copy(self.parents)
        start_index = 0

        active_assignments = []

        # iterate over different assignment types
        for ass_type in assignments:
            sources, targets, scores, ind_scores = ass_type

            num_assignments = len(scores)

            # compute selection mask
            end_index = start_index + num_assignments
            sel_mask = mask[start_index:end_index]

            # repeat the sources of active assignments
            rep_sources = np.repeat(sources[sel_mask].flatten(), targets.shape[1])

            # update the new parent array
            new_parents[targets[sel_mask].flatten()] = rep_sources

            # move selection mask indices
            start_index += num_assignments

            active_assignments += [
                (
                    sources[sel_mask],
                    targets[sel_mask],
                    scores[sel_mask],
                    ind_scores[sel_mask],
                )
            ]

        assert start_index == len(mask)

        # return the new tracking
        return SimpleTracking(
            new_parents,
            active_assignments=self.active_assignments + [active_assignments],
            parent_id=self.id,
        )

    def createIndexTracking(self):
        graph = DiGraph()

        for i, p in enumerate(self.parents):
            if p >= 0:
                graph.add_edge(p, i)

        return graph

    def createGraphTracking(self, detections):
        """
        convert the simple tracking to networkx graph containing the detections as node
        """
        graph = DiGraph()

        np_detections = np.array(detections, dtype=np.object)

        graph.add_nodes_from(np_detections[np.where(self.parents >= -1)])

        indices = np.arange(len(self.parents))
        mask = self.parents >= 0
        sel_indices = indices[mask]
        edges = np.stack([self.parents[sel_indices], sel_indices], axis=-1)

        graph.add_edges_from(np_detections[edges])

        return graph


class SimpleCluster:
    """
    Cluster groups several particles with the same tracking

    This allows more efficient inference
    """

    def __init__(
        self, tracking: SimpleTracking, size: int, cumulated_prob: Probability
    ):
        self.tracking = tracking
        self.size = size
        assert isinstance(cumulated_prob, Probability)
        self.cumulated_prob = cumulated_prob


@ray.remote
def simpleExpansionRay(
    cluster,
    sources,
    targets,
    assignment_generators,
    cutOff,
    max_num_solutions,
    pool_gap,
):
    return simpleExpansion(
        cluster,
        sources,
        targets,
        assignment_generators,
        cutOff,
        max_num_solutions,
        pool_gap=pool_gap,
    )


@ray.remote
def singleSolutionRay(
    cluster: SimpleCluster,
    sources: npt.ArrayLike[np.uint32],
    targets: npt.ArrayLike[np.uint32],
    assignment_generators: list,
    index=int,
):
    return index, singleSolution(cluster, sources, targets, assignment_generators)


def generateAssignments(
    cluster: SimpleCluster, sources, targets, assignment_generators: list
):
    """
    Generates and scores all the possible assignments from sources to targets

    sources: index list of possible sources
    targets: index list of possible targets
    assignment_generators: list of assignment generators
    """
    all_assignments = []

    # compute all possible assignments and compute their costs
    for ass_gen in assignment_generators:
        source_index, target_index, scores, individual_scores = ass_gen.generate(
            cluster.tracking, sources, targets
        )
        all_assignments.append((source_index, target_index, scores, individual_scores))

    all_assignments = list(zip(*all_assignments))

    return all_assignments


def buildMIPProblem(all_assignments, cutOff=None, solver_name="GRB"):
    """
    builds the integer program from set of assignments

    all_assignments: list of different types of assignments
    cutOff: log value of solution search cutOff
    """

    if solver_name == "GRB":
        # build MIP solver using gurobipy
        solver = SimpleGurobiSolver(all_assignments)
    else:
        # build mip solver using mip (pypi library)
        solver = SimpleMIPSolver(all_assignments, solver_name=solver_name)

    if cutOff:
        solver.setCutOff(cutOff)

    return solver


def singleSolution(
    cluster: SimpleCluster,
    sources,
    targets,
    assignment_generators: list,
    mip_method="auto",
):
    """
    Computes the best solution for an expansion

    cluster: the cluster containing the tracking until now
    sources: source index list
    targets: target index list
    assignment_generators: list of assignment generators
    """

    # generate assignments
    all_assignments = generateAssignments(
        cluster, sources, targets, assignment_generators
    )

    # build integer linear problem
    solver = buildMIPProblem(all_assignments, solver_name=mip_method)

    # solve for single solution
    max_sol = solver.solve()

    return None, max_sol


def simpleExpansion(
    cluster: SimpleCluster,
    sources,
    targets,
    assignment_generators: list,
    cutOff=None,
    max_num_solutions=50,
    pool_gap=None,
    mip_method="auto",
) -> list[tuple[SimpleCluster, Probability]]:
    """
    Computes an expansion step for a cluster

    cluster: cluster to use
    sources: df restricted table containing only source detections
    targets: df restricted table containing only target detections
    assignment_generators: list of assignment generators
    """
    # compute all possible assignments and compute their costs
    start = time.time()
    all_assignments = generateAssignments(
        cluster, sources, targets, assignment_generators
    )
    end = time.time()
    ass_gen_dur = end - start

    # compute cheapest tracking solutions with gurobi
    start = time.time()
    solver = buildMIPProblem(all_assignments, cutOff, solver_name=mip_method)
    all_solutions = solver.populate(
        max_num_solutions=max_num_solutions, pool_gap=pool_gap
    )
    end = time.time()

    # print('Num solutions', len(all_solutions))

    opt_solve_dur = end - start

    # print('ass_gen', ass_gen_dur, 'opt_solve', opt_solve_dur)

    # this is our truncated distribution proposal distribution from (9)
    sol_probabilities = [Probability(log_prob=sol[1]) for sol in all_solutions]

    if len(sol_probabilities) == 0:
        logging.warning("No expansions found")
        return []

    sampled_sol_indices = sample_from_probabilities(sol_probabilities, cluster.size)

    # According to (8) from UAT we have to sum over the full distribution
    weight = sum(
        sol_probabilities
    )  # this weight is for a single particle, multiply by the size of the cluster

    # group them to clusters and append to a new cluster list
    unique, counts = np.unique(sampled_sol_indices, return_counts=True)
    cluster_expansions = zip(unique, counts)
    new_cluster_candidates: list[tuple[SimpleCluster, float]] = []
    for solution_index, size in cluster_expansions:
        c = SimpleCluster(
            cluster.tracking.expand(all_solutions[solution_index], all_assignments),
            size,
            cumulated_prob=cluster.cumulated_prob * sol_probabilities[solution_index],
        )
        new_cluster_candidates.append((c, weight * size))

    logging.info(
        f"Expansion Step {ass_gen_dur + opt_solve_dur:.2f}s (assgen: {ass_gen_dur:.2f}s, solve: {opt_solve_dur:.2f}s)"
    )

    return new_cluster_candidates


def computeBestExpansions(
    current_cluster_dist,
    sources,
    targets,
    assignment_generators,
    num_cores,
    report_progress,
    use_ray,
    mip_method="auto",
):
    single_solutions = []

    # decide parallelized execution
    if num_cores == 1:
        # iterate over clusters and expand one by one
        it = current_cluster_dist

        if report_progress:
            it = tqdm(current_cluster_dist, leave=False)

        for cluster in it:
            single_solutions.append(
                singleSolution(
                    cluster,
                    sources,
                    targets,
                    assignment_generators,
                    mip_method=mip_method,
                )
            )
    elif use_ray:
        # prepare ray
        sources_remote = ray.put(sources)
        targets_reomte = ray.put(targets)
        assignment_generators_reomte = ray.put(assignment_generators)
        # perform computation
        single_solutions = [
            singleSolutionRay.remote(
                cluster=c,
                sources=sources_remote,
                targets=targets_reomte,
                assignment_generators=assignment_generators_reomte,
                index=i,
            )
            for i, c in enumerate(current_cluster_dist)
        ]

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        ray_it = to_iterator(
            single_solutions
        )  # ray.get(new_cluster_candidates_and_weights)

        if report_progress:
            ray_it = tqdm(ray_it, total=len(single_solutions), leave=False)

        split_solutions = list(zip(*list(ray_it)))

        logging.info(split_solutions[0])
        order = np.argsort(np.array(list(split_solutions[0]), dtype=np.int32))

        # reorder single solutions so that they match the cluster list (this is necessary due to multiprocessing)
        single_solutions = np.array(list(split_solutions[1]), dtype=np.object)[
            order
        ]  # reduce(lambda a,b: a+b, ray_it)
    else:
        # use multiprocessing for parallelization
        with multiprocessing.Pool(num_cores) as p:
            single_solutions = list(
                p.map(
                    lambda c: singleSolution(
                        c, sources, targets, assignment_generators
                    ),
                    current_cluster_dist,
                    chunksize=4,
                )
            )

    # return
    return single_solutions


def computeAllExpansionsAdv(
    current_cluster_dist,
    sources,
    targets,
    assignment_generators,
    num_cores,
    report_progress,
    use_ray,
    truncation_threshold=np.log(1e-3),
    max_num_solutions=50,
    mip_method="auto",
):
    # compute all best expansions
    best_expansions = computeBestExpansions(
        current_cluster_dist,
        sources,
        targets,
        assignment_generators,
        num_cores,
        report_progress,
        use_ray,
        mip_method=mip_method,
    )

    # compute the maximum likely solution
    best_exps_zipped = list(zip(*best_expansions))
    maxSols = np.array(list(map(lambda sol: sol[1], best_exps_zipped[1])))

    # get maximum possible cumulated probability
    cumLogProbs = np.array(
        [c.cumulated_prob.log_probability for c in current_cluster_dist]
    )
    bestCumSol = np.max(maxSols + cumLogProbs)

    # best possible cumulated probability for every cluster (cluster cumProb * expansion prob)
    individualBestCumLogProbs = maxSols + cumLogProbs

    # compute truncation threshold based on that
    truncation = truncation_threshold
    cutOff = bestCumSol + truncation
    truncation_mask = individualBestCumLogProbs < cutOff
    pruned_clusters = np.sum(truncation_mask)

    logging.info(f"Possible pruned clusters: {pruned_clusters}")

    # for the solution pool we must compute the relative gap between the best solution and the worst solution: worst = best + gap * best
    individual_pool_gaps = (
        cutOff - individualBestCumLogProbs
    ) / individualBestCumLogProbs

    # pruning cluster distributions and pool gaps
    pruned_clusters_dist = np.array(current_cluster_dist)[~truncation_mask]
    individual_pool_gaps = individual_pool_gaps[~truncation_mask]

    logging.info(individual_pool_gaps)

    return computeAllExpansions(
        pruned_clusters_dist,
        sources,
        targets,
        assignment_generators,
        num_cores,
        report_progress,
        use_ray,
        cutOff=cutOff,
        max_num_solutions=max_num_solutions,
        individual_pool_gaps=individual_pool_gaps,
        mip_method=mip_method,
    )


def computeAllExpansions(
    current_cluster_dist,
    sources,
    targets,
    assignment_generators,
    num_cores,
    report_progress,
    use_ray,
    individual_pool_gaps,
    cutOff=None,
    max_num_solutions=50,
    mip_method="auto",
):
    new_cluster_candidates_and_weights = []

    # decide parallelized execution
    if num_cores == 1:
        # iterate over clusters and expand one by one
        it = current_cluster_dist

        if report_progress:
            it = tqdm(current_cluster_dist, leave=False)

        for i, cluster in enumerate(it):
            new_cluster_candidates_and_weights += simpleExpansion(
                cluster,
                sources,
                targets,
                assignment_generators,
                cutOff,
                max_num_solutions=max_num_solutions,
                pool_gap=individual_pool_gaps[i],
                mip_method=mip_method,
            )
    elif use_ray:
        # prepare ray
        sources_remote = ray.put(sources)
        targets_reomte = ray.put(targets)
        assignment_generators_reomte = ray.put(assignment_generators)
        # perform computation
        new_cluster_candidates_and_weights = [
            simpleExpansionRay.remote(
                cluster=c,
                sources=sources_remote,
                targets=targets_reomte,
                assignment_generators=assignment_generators_reomte,
                cutOff=cutOff,
                max_num_solutions=max_num_solutions,
                pool_gap=individual_pool_gaps[i],
            )
            for i, c in enumerate(current_cluster_dist)
        ]

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        ray_it = to_iterator(
            new_cluster_candidates_and_weights
        )  # ray.get(new_cluster_candidates_and_weights)

        if report_progress:
            ray_it = tqdm(
                ray_it, total=len(new_cluster_candidates_and_weights), leave=False
            )

        new_cluster_candidates_and_weights = reduce(lambda a, b: a + b, ray_it)
    else:
        # use multiprocessing for parallelization
        with multiprocessing.Pool(num_cores) as p:
            new_cluster_candidates_and_weights += reduce(
                lambda a, b: a + b,
                p.map(
                    lambda c: simpleExpansion(
                        c[1],
                        sources,
                        targets,
                        assignment_generators,
                        cutOff,
                        max_num_solutions=max_num_solutions,
                        pool_gap=individual_pool_gaps[c[0]],
                    ),
                    enumerate(current_cluster_dist),
                    chunksize=4,
                ),
                [],
            )

    return new_cluster_candidates_and_weights


def simpleTracking(
    df: DataFrame,
    assignment_generators,
    num_particles,
    num_cores,
    use_ray=True,
    reporters=None,
    report_progress=True,
    max_num_hypotheses=int(1e3),
    cutOff=np.log(1e-3),
    max_num_solutions=50,
    saving_interval=500,
    mip_method="auto",
    frames=None,
):
    """
    df: data frame containing all cell information
    assignment_generators: list of assignment generators
    num_particles: number of particles for the particle filter
    num_cores: number of execution cores to be used
    use_ray: if true and more than one core should be used, ray package is used for parallelization otherwise multiprocessing is used
    """

    if reporters is None:
        reporters = []

    # try to launch ray (only if really needed be)
    if num_cores > 1 and use_ray and not ray.is_initialized():
        ray.init(num_cpus=num_cores)

    # get the frame list
    if frames is None:
        frames = np.unique(df["frame"])

    logging.info(f"Frames: {frames}")

    # create initial empty cluster with full size and probability 1
    current_cluster_dist = [
        SimpleCluster(
            SimpleTracking(
                np.ones((len(df),), dtype=np.int32) * -2, parent_id=-1, id=0
            ),
            size=num_particles,
            cumulated_prob=Probability(1),
        )
    ]
    next_tracking_id = 1
    # all_clusters = [current_cluster_dist]

    try:
        # loop over consecutive frames
        for frame_index, (source_frame, target_frame) in enumerate(
            tqdm(
                zip(frames, frames[1:]), total=len(frames) - 1, desc="Perform tracking"
            )
        ):
            logging.info(f"Frames: {source_frame} -> {target_frame}")

            # find source and target detections
            sources = df[df["frame"] == source_frame].index.to_numpy(
                dtype=np.int32
            )  # TODO find only trackends
            targets = df[df["frame"] == target_frame].index.to_numpy(dtype=np.int32)

            logging.info(f"trackEnds {len(sources)} --> {(len(targets))} detections")

            new_cluster_candidates_and_weights: list[
                tuple[SimpleCluster, Probability]
            ] = []

            logging.info("Num clusters: {len(current_cluster_dist)}")

            new_cluster_candidates_and_weights = computeAllExpansionsAdv(
                current_cluster_dist,
                sources,
                targets,
                assignment_generators,
                num_cores,
                report_progress,
                use_ray,
                truncation_threshold=cutOff,
                max_num_solutions=max_num_solutions,
                mip_method=mip_method,
            )

            # unpack the new cluster candidates with their resampling weights
            new_cluster_candidates, resampling_weights = zip(
                *new_cluster_candidates_and_weights
            )

            assert len(new_cluster_candidates) > 0, "No expansions solutions found!"

            # resample
            sampled_cluster_indices = sample_from_probabilities(
                resampling_weights, num_particles
            )
            unique, count = np.unique(sampled_cluster_indices, return_counts=True)
            histogram = zip(unique, count)

            # choose sampled clusters and correct their size
            new_clusters = []
            for cluster_index, size in histogram:
                cluster = copy.deepcopy(new_cluster_candidates[cluster_index])
                cluster.size = size
                cluster.tracking.id = next_tracking_id
                next_tracking_id += 1
                new_clusters.append(cluster)

            # update cluster distribution
            current_cluster_dist = new_clusters

            # sort clusters w.r.t. size (descending)
            current_cluster_dist = sorted(
                current_cluster_dist, key=lambda c: c.size, reverse=True
            )

            # cut the number of hypothesis
            current_cluster_dist = current_cluster_dist[:max_num_hypotheses]

            for i, c in enumerate(current_cluster_dist):
                c.tracking.id = i

            # all_clusters += [current_cluster_dist]

            for r in reporters:
                r.report_distribution(current_cluster_dist)

            if frame_index % saving_interval == 0 and frame_index > 0:
                for r in reporters:
                    r.close()

    finally:

        for r in reporters:
            r.close()

    return current_cluster_dist


def split_pair_filter(
    source_index, target_index, cdc: ContourDistanceCache, distance_threshold=1.0
):
    # pylint: disable=unused-argument

    return cdc.distance(target_index[:, 0], target_index[:, 1]) < distance_threshold


class SimpleTrackingReporter:
    """Reporting tracking result in simple format"""

    def __init__(
        self, output_folder: str, df: DataFrame, detections, assignment_generators
    ):
        self.all_clusters = []
        self.output_folder = output_folder
        self.df = df
        self.detections = detections
        self.timings = []
        self.timings.append(time.time())
        self.assignment_generators = assignment_generators

        self.final_cluster = None

    def report_distribution(self, current_clusters: list[SimpleCluster]):
        self.final_cluster = current_clusters[0]

    def close(self):
        Path(self.output_folder).mkdir(exist_ok=True)
        edge_list = list(
            map(
                lambda e: (int(e[0]), int(e[1])),
                self.final_cluster.tracking.createIndexTracking().edges,
            )
        )

        tracking_data = [
            dict(
                sourceId=self.detections[edge[0]].id,
                targetId=self.detections[edge[1]].id,
            )
            for edge in edge_list
        ]
        segmentation_data = [
            dict(
                label=cont.label,
                contour=cont.coordinates.tolist(),
                id=cont.id,
                frame=cont.frame,
            )
            for cont in self.detections
        ]

        data_structure = dict(
            segmentation=segmentation_data,
            tracking=tracking_data,
            format_version="0.0.1",
        )

        with open(
            Path(self.output_folder) / "tracking.json", "w", encoding="utf-8"
        ) as output_file:
            json.dump(data_structure, output_file)
        logging.info(self.final_cluster.tracking.createIndexTracking().edges)
