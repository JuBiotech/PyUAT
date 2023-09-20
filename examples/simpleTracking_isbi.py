""" Code for the framerate comparison """

import gzip
import sys
import time
from pathlib import Path

import numpy as np
from acia.tracking.formats import parse_simple_tracking
from pandas import DataFrame

from uat.core import simpleTracking
from uat.output import SimpleTrackingReporter

from .isbi_config import setup_assignment_generators
from .utils import compute_axes_info


def main(source_file, output_file, subsampling):

    source_file = Path(source_file)
    output_file = Path(output_file)
    subsampling = int(subsampling)

    ### load data
    with open(source_file, encoding="UTF-8") as input_file:
        ov, _ = parse_simple_tracking(input_file.read())

    ### load config
    ### perform tracking

    all_detections = ov.contours

    # TODO: hardcoded
    width = 434

    # split_proposals, major_axes = compute_split_proposals(all_detections)

    # extract arguments
    num_particles = 1  # args.nb_particles
    num_cores = 1  # args.nb_cpus
    max_num_hypotheses = 1  # args.nb_max_hypotheses
    cutOff = -1  # args.cutOff
    max_num_solutions = 1  # args.sol_pool_size

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
        ) = compute_axes_info(det)

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
    print(df)

    # put data into numpy arrays (greatly increases the speed, as data can be immediately indexed)
    data = {
        "area": np.array(df["area"].to_list(), dtype=np.float32),
        "centroid": np.array(df["centroid"].to_list(), dtype=np.float32),
        "major_extents": np.array(df["major_extents"].to_list(), dtype=np.float32),
    }

    print(df)

    assignment_generators = setup_assignment_generators(df, data, width, subsampling)

    # create reporters
    reporters = [
        SimpleTrackingReporter(
            output_file.parent, df, all_detections, assignment_generators
        )
    ]

    print(reporters[0].output_folder)

    # start tracking
    start = time.time()
    simpleTracking(
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

    ### export output

    with gzip.open(output_file.parent / "tracking.json.gz", "r") as input_file:
        with open(output_file, "wb") as out_f:
            out_f.write(input_file.read())


if __name__ == "__main__":
    sf, of, sub = (sys.argv[1], sys.argv[2], sys.argv[3])
    main(sf, of, sub)
