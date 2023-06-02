""" Example for tracking an image sequence from omero where the segmentation is already available as gzipped json"""

from __future__ import annotations

import gzip
import time
from pathlib import Path

# standard configuration for gurobi
import gurobipy as gp
import numpy as np
from acia.base import Contour, Overlay
from acia.segm.formats import parse_simple_segmentation

# pylint: disable=unused-import
from acia.segm.omero.storer import (
    OmeroSequenceSource,
    download_file_from_object,
    replace_file_annotation,
)
from acia.segm.omero.utils import getImage
from config import setup_assignment_generators
from credentials import host, password, port, username
from gurobipy import GRB
from networkx import DiGraph
from omero.gateway import BlitzGateway
from pandas import DataFrame
from utils import cluster_to_tracking_graph, render_tracking_video

from uat.core import simpleTracking
from uat.output import SimpleTrackingReporter

gp.setParam(GRB.Param.LogFile, "gurobi.log")
gp.setParam(GRB.Param.LogToConsole, 0)
gp.setParam(GRB.Param.TimeLimit, 300)


def main(output_folder=Path("tracking_output"), omero_id=18001):

    output_folder.mkdir(exist_ok=True, parents=True)

    # name of the omero file attachment
    source_file = "pred_simpleSegmentation.json.gz"

    print(f"Perform tracking for {omero_id}...")

    # download segmentation from omero and retrieve width and height
    with BlitzGateway(username, password, host=host, port=port, secure=True) as conn:
        print("Downloading segmentation...")
        download_file_from_object(
            "Image", omero_id, source_file, output_folder / source_file, conn
        )

        image = getImage(conn, omero_id)

        width = image.getSizeX()
        # height = image.getSizeY()
        end_frame = image.getSizeT() - 1

    start_frame = 0
    # end_frame = 30  # 1600 #1599 #2564

    # width, height = (997, 1050)

    def is_close(cont, min_dist=3):
        minx, _ = np.min(cont.coordinates, axis=0)
        maxx, _ = np.max(cont.coordinates, axis=0)

        return minx < min_dist or maxx > width - min_dist

    print("Loading segmentation...")
    # Load segmentation or additional tracking information
    with gzip.open(output_folder / source_file) as input_file:
        ov = parse_simple_segmentation(input_file.read())

    # remove close to border predictions (not fully observed cells behave strangely for growth, position, ...)
    ov = Overlay(
        [
            cont
            for cont in ov
            if not is_close(cont) and start_frame <= cont.frame <= end_frame
        ],
        range(start_frame, end_frame + 1),
    )

    # create the timed iterator
    ors = ov.timeIterator(startFrame=start_frame, endFrame=end_frame)

    all_detections: list[Contour] = [cont for overlay in ors for cont in overlay]
    print(len(all_detections))

    # split_proposals, major_axes = compute_split_proposals(all_detections)

    # extract arguments
    num_particles = 1  # args.nb_particles
    num_cores = 1  # args.nb_cpus
    max_num_hypotheses = 1  # args.nb_max_hypotheses
    cutOff = -1  # args.cutOff
    max_num_solutions = 1  # args.sol_pool_size

    # create the main data frame
    df = DataFrame(
        [
            {
                "area": det.area,
                "centroid": np.array(det.center),
                "perimeter": det.polygon.length,
                "id": det.id,
                "frame": det.frame,
                "contour": np.array(det.coordinates),
            }
            for i, det in enumerate(all_detections)
        ]
    )

    # put data into numpy arrays (greatly increases the speed, as data can be immediately indexed)
    data = {
        "area": np.array(df["area"].to_list(), dtype=np.float32),
        "centroid": np.array(df["centroid"].to_list(), dtype=np.float32),
    }

    print(df)

    assignment_generators = setup_assignment_generators(df, data, width)

    # create reporters
    reporters = [
        SimpleTrackingReporter(output_folder, df, all_detections, assignment_generators)
    ]

    # start tracking
    start = time.time()
    last_cluster_dist = simpleTracking(
        df,
        assignment_generators,
        num_particles,
        num_cores=num_cores,
        reporters=reporters,
        max_num_hypotheses=max_num_hypotheses,
        cutOff=cutOff,
        max_num_solutions=max_num_solutions,
    )
    end = time.time()

    print("time for tracking", end - start)

    # converte simple cluster to tracking graph
    tracking = cluster_to_tracking_graph(last_cluster_dist[0], ov)

    # render the tracking video
    oss = OmeroSequenceSource(omero_id, username, password, host, port)
    render_tracking_video(oss, ov, tracking, output_file=output_folder / "tracking.avi")

    # TODO: upload complete tracking


if __name__ == "__main__":
    main()
