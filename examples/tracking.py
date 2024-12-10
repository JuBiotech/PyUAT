""" Example for tracking an image sequence from omero where the segmentation is already available as gzipped json"""

from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

# standard configuration for gurobi
import gurobipy as gp
import numpy as np
from acia.base import Contour, Overlay
from acia.segm.formats import parse_simple_segmentation

# pylint: disable=unused-import
from acia.segm.omero.storer import download_file_from_object
from acia.segm.omero.utils import getImage
from config import setup_assignment_generators
from credentials import host, password, port, username
from gurobipy import GRB
from omero.gateway import BlitzGateway

from uat.core import simpleTracking
from uat.utils import load_single_cell_information, save_tracking

gp.setParam(GRB.Param.LogFile, "gurobi.log")
gp.setParam(GRB.Param.LogToConsole, 0)
gp.setParam(GRB.Param.TimeLimit, 300)


def load_omero_data(output_folder, omero_id, subsampling_factor=40):
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
    end_frame = 799  # 1600 #1599 #2564

    frames = set(np.arrange(start_frame, end_frame + 1, subsampling_factor))

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
        [cont for cont in ov if cont.frame in frames and not is_close(cont)],
        frames=frames,
    )

    # create the timed iterator
    ors = ov.timeIterator(startFrame=start_frame, endFrame=end_frame)

    all_detections: list[Contour] = [cont for overlay in ors for cont in overlay]
    print(len(all_detections))

    return all_detections, width, ov


def main(input_file, output_file):

    # extract arguments
    num_particles = 1  # args.nb_particles
    num_cores = 1  # args.nb_cpus
    max_num_hypotheses = 1  # args.nb_max_hypotheses
    cutOff = -1  # -10  # args.cutOff
    max_num_solutions = 1  # 10  # args.sol_pool_size

    df, all_detections = load_single_cell_information(input_file)

    # TODO: hardcoded
    width = 434

    assignment_generators = setup_assignment_generators(df, width)

    # start tracking
    start = time.time()
    res = simpleTracking(
        df,
        assignment_generators,
        num_particles,
        num_cores=num_cores,
        max_num_hypotheses=max_num_hypotheses,
        cutOff=cutOff,
        max_num_solutions=max_num_solutions,
        mip_method="auto",  # use gurobi if it is installed, otherwise go back to CBC (slower)
    )
    end = time.time()

    print("time for tracking", end - start)

    save_tracking(res[0], all_detections, output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="PyUAT",
        description="Tracking of single-cells in timelapse sequences",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        default="examples/pred_simpleSegmentation.json",
    )
    parser.add_argument("-o", "--output_file", type=str, default="tracking.json.gz")

    args = parser.parse_args()

    main(Path(args.input_file), Path(args.output_file))