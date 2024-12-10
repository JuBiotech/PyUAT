""" Collection of utility functionality """

from __future__ import annotations

import gzip
import itertools
import json
import multiprocessing
from functools import partial, reduce
from itertools import tee
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import ray
from acia.base import Overlay
from acia.segm.formats import parse_simple_segmentation
from acia.tracking.formats import parse_simple_tracking
from acia.tracking.utils import subsample_lineage
from acia.viz import VideoExporter2
from pandas import DataFrame
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def product(list1):
    return reduce((lambda x, y: x * y), list1)


def distance(point_a, point_b):
    """
    point_a: (x,y, ...)
    point_b: (x,y, ...)
    """

    assert len(point_a) == len(point_b)

    return np.linalg.norm(np.array(point_a) - np.array(point_b), ord=2)


class DistanceComputer:
    """Compute and cache distances"""

    def __init__(self):
        """
        Precompute all possible distances
        """
        self._distance_lookup = {}

    def precompute(self, detections, num_cores=1):
        if num_cores == 1:
            self.precomputeDistances(detections)
        else:
            self.precomputeDistancesParallel(detections, num_cores)

    def precomputeDistances(self, detections):
        self._distance_lookup = {}
        detections = list(detections)
        max_frame = max(map(lambda det: det.frame_id, detections))
        results = itertools.chain.from_iterable(
            map(
                lambda frame_id: DistanceComputer.precomputeDistancesInFrame(
                    detections, frame_id
                ),
                range(0, max_frame + 1),
            )
        )
        for key, value in results:
            self._distance_lookup[key] = value

    @staticmethod
    def precomputeDistancesInFrame(detections, frame_id):
        result = []
        frame_detections = filter(lambda det: det.frame_id == frame_id, detections)
        for det1, det2 in itertools.combinations(frame_detections, 2):
            result.append(
                (tuple(sorted([det1.id, det2.id])), det1.polygon.distance(det2.polygon))
            )

        return result

    @staticmethod
    def distance_comp(det_set):
        assert len(det_set) == 2
        det1, det2 = det_set[0], det_set[1]
        return (tuple(sorted([det1.id, det2.id])), det1.polygon.distance(det2.polygon))

    def precomputeDistancesParallel(self, detections, num_cores):
        self._distance_lookup = {}
        detections = list(detections)
        max_frame = max(map(lambda det: det.frame_id, detections))
        with multiprocessing.Pool(num_cores) as p:

            result = []
            for ind_result in tqdm(
                p.imap_unordered(
                    lambda frame_id: DistanceComputer.precomputeDistancesInFrame(
                        detections, frame_id
                    ),
                    reversed(range(0, max_frame + 1)),
                )
            ):
                result.append(ind_result)

            p.close()
            p.join()
            p.terminate()

        # result = p.map(DistanceComputer.distance_comp, itertools.combinations(detections, 2))

        for key, value in itertools.chain.from_iterable(result):
            self._distance_lookup[key] = value

    def distance(self, det1, det2):
        key = tuple(sorted([det1.id, det2.id]))
        if key not in self._distance_lookup:
            self._distance_lookup[key] = det1.polygon.distance(det2.polygon)
        return self._distance_lookup[key]


def pair_ind_to_dist_ind(d, i, j):
    assert np.all(i < j)
    index = d * (d - 1) / 2 - (d - i) * (d - i - 1) / 2 + j - i - 1
    return index.astype(np.uint32)


def compute_distances(i, contours):
    local_distances = []

    for j in range(i + 1, len(contours)):
        local_distances.append(np.min(cdist(contours[i], contours[j])))

    return local_distances


@ray.remote
def ray_compute_distances(i, contours):
    return compute_distances(i, contours)


class ContourDistanceCache:
    """Cache for quickly computing the contour distance cache"""

    def __init__(self, df):
        self.n = len(df)

        contours = df["contour"].to_numpy()

        distances = []
        distances = self.__multi_processing(contours)
        # with multiprocessing.Pool(16) as pool:
        #    distances = list(tqdm(pool.imap(partial(compute_distances(contours=contours)),range(self.n)), total=self.n))
        #    #for i in tqdm(range(self.n)):

        self.distances = np.concatenate(distances)

        print(self.distances.shape)

    def __multi_processing(self, contours):
        return process_map(
            partial(compute_distances, contours=contours), range(self.n), chunksize=10
        )

    def __ray_processing(self, contours):
        # pylint: disable=unused-private-member
        distances = [ray_compute_distances.remote(i, contours) for i in range(self.n)]

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        ray_it = to_iterator(distances)

        # if report_progress:
        ray_it = tqdm(ray_it, total=len(distances))

        return list(ray_it)

        # return ray_compute_distances()

    def distance(self, source_index, target_index):
        return self.distances[pair_ind_to_dist_ind(self.n, source_index, target_index)]


class NearestNeighborCache:
    """Cache for fast computation of nearest neighbor sets"""

    def __init__(self, df):
        self.df = df

        self.positions = np.array(df["centroid"].tolist(), dtype=np.float32)

    @property
    def num_elements(self):
        return len(self.positions)

    def computePossibleTargets(self, targetFrame: int):
        # first filter data frame for frame targets
        return self.df[self.df["frame"] == targetFrame].index.to_numpy()

    def neighborDistances(self, listOfSources, possible_targets):

        # repeat the targets for the number of sources
        stacked_targets = np.tile(possible_targets, (len(listOfSources),))
        # repeat the sources for the number of targets
        stacked_sources = np.repeat(listOfSources, len(possible_targets))

        # print(stacked_sources, stacked_targets)

        # compute the right indices
        # indices = pair_ind_to_dist_ind(self.num_elements, stacked_sources, stacked_targets)

        # lookup the distances
        # return self.distanceCache.pairwise_distance[indices].reshape(-1, len(possible_targets))
        return self.distance(stacked_sources, stacked_targets).reshape(
            -1, len(possible_targets)
        )

    def neighborDistancesMatrix(self, sourceIndices, targetIndexMatrix):
        num_targets = targetIndexMatrix.shape[1]

        # repeat the sources for the number of targets
        stacked_sources = np.repeat(sourceIndices, num_targets)
        flattenedTargets = targetIndexMatrix.flatten()

        # compute the right indices
        # indices = pair_ind_to_dist_ind(self.num_elements, stacked_sources, flattenedTargets)

        # lookup the distances
        return self.distance(stacked_sources, flattenedTargets).reshape(-1, num_targets)

    def sortedNeighbors(self, listOfSources, targetFrame):
        """
        listOfSources: n source indices
        targetFrame: int

        returns nxk index map where k is the number of detections in the target frame
        """
        # identify all detections in target frame
        possible_targets = self.computePossibleTargets(targetFrame)

        distances = self.neighborDistances(listOfSources, possible_targets)

        # sort them by indices
        return possible_targets[np.argsort(distances).flatten()].reshape(
            len(listOfSources, len(possible_targets))
        )

    def kNearestNeighborsFrame(self, k: int, listOfSources, targetFrame):
        # identify all detections in target frame
        possible_targets = self.computePossibleTargets(targetFrame)

        if len(possible_targets) <= k:
            # if we have lesseq than k detections finding k-nearest neighbors is trivial
            return np.tile(possible_targets, (len(listOfSources),)).reshape(
                (-1, len(possible_targets))
            )

        # compute distances
        distances = self.neighborDistances(listOfSources, possible_targets)

        # print(distances)

        # sort them by indices
        return possible_targets[np.argpartition(distances, k)[:, :k].flatten()].reshape(
            (len(listOfSources), k)
        )

    def kNearestNeigbors(self, k: int, sourceIndices, targetIndices):
        distances = self.neighborDistancesMatrix(sourceIndices, targetIndices)

        # print('dist shape', distances.shape)

        return np.argpartition(distances, k)[:, :k]  # .reshape((len(listOfSources), k))

    def kNearestNeigborsMatrixMask(self, k: int, sourceIndices, targetIndexMatrix):
        if targetIndexMatrix.shape[1] <= k:
            # if we have lesseq than k detections finding k-nearest neighbors is trivial
            mask = np.ones_like(targetIndexMatrix, dtype=bool)

        else:
            correctIndices = self.kNearestNeigbors(k, sourceIndices, targetIndexMatrix)

            # print(correctIndices.shape)

            # convert the filtered index list into a mask on the full target index matrix
            mask = np.zeros_like(targetIndexMatrix)
            # writing the true values is bit complex due to selection
            mask[np.arange(sourceIndices.shape[0])[:, None], correctIndices] = True

        return mask

    def distance(self, indexListA, indexListB):
        distances = np.linalg.norm(
            self.positions[indexListA] - self.positions[indexListB], axis=-1
        )
        return distances
        # return self.distanceCache.pairwise_distance[pair_ind_to_dist_ind(self.num_elements, indexListA, indexListB)]


def load_local_data(simple_seg_file):
    print("Loading segmentation...")
    # Load segmentation or additional tracking information
    with open(simple_seg_file, encoding="UTF-8") as input_file:
        ov = parse_simple_segmentation(input_file.read())

    return ov


def load_single_cell_information(input_file: Path) -> DataFrame:

    # all_detections, width, ov = load_local_data(output_folder/ "pred_simpleSegmentation.json.gz", width=997, subsampling_factor=40) #load_omero_data(output_folder, omero_id)
    ov = load_local_data(input_file)
    # ov = Overlay([cont for cont in ov if cont.frame <= 100])
    all_detections = ov.contours

    # split_proposals, major_axes = compute_split_proposals(all_detections)

    # create the main data frame
    entries = []
    for _, det in enumerate(all_detections):
        # compute the axis info
        (
            width,
            length,
            major_axis,
            minor_axis,
            major_extents,
            minor_extents,
        ) = compute_axes_info(det.polygon)

        # add entry for this cell
        entries.append(
            {
                "area": det.area,
                "centroid": np.array(det.center),
                "perimeter": det.polygon.length,
                "id": det.id,
                "frame": det.frame,
                "contour": np.array(det.coordinates),
                "width": width,
                "length": length,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "major_extents": major_extents,
                "minor_extents": minor_extents,
            }
        )
    df = DataFrame(entries)

    return df, all_detections


def extract_single_cell_information(ov: Overlay) -> DataFrame:

    # ov = Overlay([cont for cont in ov if cont.frame <= 100])
    all_detections = ov.contours

    # split_proposals, major_axes = compute_split_proposals(all_detections)

    # create the main data frame
    entries = []
    for _, det in enumerate(all_detections):
        # compute the axis info
        (
            width,
            length,
            major_axis,
            minor_axis,
            major_extents,
            minor_extents,
        ) = compute_axes_info(det.polygon)

        # add entry for this cell
        entries.append(
            {
                "area": det.area,
                "centroid": np.array(det.center),
                "perimeter": det.polygon.length,
                "id": det.id,
                "frame": det.frame,
                "contour": np.array(det.coordinates),
                "width": width,
                "length": length,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "major_extents": major_extents,
                "minor_extents": minor_extents,
            }
        )
    df = DataFrame(entries)

    return df, all_detections


def compute_axes_info(polygon):
    """Extract information about the major and minor axes from a single detection

    Args:
        det (_type_): detection object

    Returns:
        _type_: a lot of information about major and minor axes
    """
    # get the minimum rotated rectangle around a detection
    mrr = polygon.minimum_rotated_rectangle
    coordinates = np.array(mrr.boundary.coords)

    # substract all coordinates to get the distances
    diff_vectors = coordinates[:-1] - coordinates[1:]
    distances = np.linalg.norm(diff_vectors, axis=1)

    # minor axis has lower distance than major axis
    minor_index = np.argmin(distances)
    major_index = np.argmax(distances)

    # compute width and length (minor axis, major axis)
    width = distances[minor_index]
    length = distances[major_index]

    # extract minor and major axis
    minor_axis = diff_vectors[minor_index]
    major_axis = diff_vectors[major_index]

    center = np.array(mrr.centroid.coords)[0]

    # construct axis endpoints (extents)
    major_extents = center + 0.5 * np.array([major_axis, -major_axis])
    minor_extents = center + 0.5 * np.array([minor_axis, -minor_axis])

    # return all extracted values
    return width, length, major_axis, minor_axis, major_extents, minor_extents


def save_tracking(final_cluster, detections, output_file: Path = "tracking.json.gz"):
    # Path(self.output_folder).mkdir(exist_ok=True)
    edge_list = list(
        map(
            lambda e: (int(e[0]), int(e[1])),
            final_cluster.tracking.createIndexTracking().edges,
        )
    )

    tracking_data = [
        dict(
            sourceId=detections[edge[0]].id,
            targetId=detections[edge[1]].id,
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
        for cont in detections
    ]

    data_structure = dict(
        segmentation=segmentation_data,
        tracking=tracking_data,
        format_version="0.0.1",
    )

    with gzip.open(output_file, "wt") as output_writer:
        json.dump(data_structure, output_writer)


def render_tracking_video(
    image_source,
    overlay_iterator,
    tracking,
    output_file="track.mp4",
    framerate=3,
):
    """Render tracking video

    Args:
        image_source (_type_): acia image source iterator
        overlay_iterator (_type_): iterator over cell overlay
        tracking (_type_): tracking graph
        output_file (str, optional): Output file for the video. Defaults to "track.avi".
        framerate (int, optional): Rendering framerate (fps). Defaults to 3.
    """

    overlay_iterator, consumer = tee(overlay_iterator)
    all_contours = reduce(
        lambda a, b: a + b,
        [[cont for overlay in iter(consumer) for cont in overlay]],
        [],
    )
    contour_lookup = {cont.id: cont for cont in all_contours}

    with VideoExporter2.fast_vp9(str(output_file), framerate) as ve:
        for frame, (image, overlay) in enumerate(
            tqdm(zip(image_source, overlay_iterator))
        ):
            pil_image = Image.fromarray(image.raw)
            overlay.draw(pil_image, "#00FFFF", None)

            draw = ImageDraw.Draw(pil_image)

            draw.text((30, 30), f"Frame: {frame}", fill="white")

            np_image = np.array(pil_image)

            for cont in overlay:
                if cont.id in tracking.nodes:
                    edges = tracking.out_edges(cont.id)

                    born = tracking.in_degree(cont.id) == 0

                    for edge in edges:
                        source = contour_lookup[edge[0]].center
                        target = contour_lookup[edge[1]].center

                        line_color = (0, 0, 255)  # bgr: red

                        if len(edges) > 1:
                            line_color = (255, 0, 0)  # bgr: blue

                        cv2.line(
                            np_image,
                            tuple(map(int, source)),
                            tuple(map(int, target)),
                            line_color,
                            thickness=3,
                        )

                        if born:
                            cv2.circle(
                                np_image,
                                tuple(map(int, source)),
                                3,
                                (203, 192, 255),
                                thickness=1,
                            )

                    if len(edges) == 0:
                        cv2.rectangle(
                            np_image,
                            cont.center.astype(np.int32) - 2,
                            cont.center.astype(np.int32) + 2,
                            (203, 192, 255),
                        )

                        # draw.line([tuple(source), tuple(target)], fill="#FF0000", width=2)

            # cv2.imwrite(str(output_folder / f"image{frame:03}.png"), np_image)
            ve.write(np_image)


def load_data(
    tracking_file: Path, subsampling_factor: int, end_frame: int
) -> tuple[Overlay, nx.DiGraph]:
    """Load simple tracking ground truth

    Args:
        tracking_file (Path): Path to json file
        subsampling_factor (int): the subsampling factor that shall be applied
        end_frame (int): Cut-off frame

    Returns:
        tuple[Overlay, nx.DiGraph]: segmentation overlay and tracking graph
    """
    with open(tracking_file, encoding="utf-8") as tr_input:
        overlay, tracking_graph = parse_simple_tracking(tr_input.read())

    sub_tracking_graph = subsample_lineage(tracking_graph, subsampling_factor)

    nds_to_delete = []
    for node in sub_tracking_graph.nodes:
        if sub_tracking_graph.nodes[node]["frame"] > end_frame:
            nds_to_delete.append(node)

    sub_tracking_graph.remove_nodes_from(nds_to_delete)

    sub_nodes = set(sub_tracking_graph.nodes)

    sub_overlay = Overlay([cont for cont in overlay if cont.id in sub_nodes])

    # correct frame number
    for cont in overlay:
        cont.frame = int(np.round(cont.frame / subsampling_factor))
    for n in tracking_graph.nodes:
        tracking_graph.nodes[n]["frame"] = int(
            np.round(tracking_graph.nodes[n]["frame"] / subsampling_factor)
        )

    return sub_overlay, sub_tracking_graph
