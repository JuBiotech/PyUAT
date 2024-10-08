""" Test tracking models
"""
import time
import unittest
from pathlib import Path

import numpy as np
import wget
from acia.base import Overlay
from acia.tracking.formats import parse_simple_tracking
from acia.tracking.utils import subsample_lineage

#  pylint: disable=unused-import
from uat.config import (
    add_angle_models,
    add_growth_model,
    make_assignmet_generators,
    use_first_order_model,
    use_nearest_neighbor,
)
from uat.core import simpleTracking
from uat.utils import extract_single_cell_information, save_tracking


def setup_assignment_generators(df, subsampling_factor: int):

    # arrange single-cell information into numpy arrays (greatly increases the speed, as data can be immediately indexed)
    data = {
        "area": np.array(df["area"].to_list(), dtype=np.float32),
        "centroid": np.array(df["centroid"].to_list(), dtype=np.float32),
        "major_extents": np.array(df["major_extents"].to_list(), dtype=np.float32),
        "major_axis": np.array(df["major_axis"].to_list(), dtype=np.float32),
    }

    # create biologically motivated models
    # constant_new_models, constant_end_models, migration_models, split_models = add_growth_model(data=data, subsampling=subsampling_factor) #add_angle_models(data=data, subsampling=subsampling_factor) #use_first_order_model(data=data, subsampling=subsampling_factor) #use_nearest_neighbor(data=data, subsampling=subsampling_factor) #
    (
        constant_new_models,
        constant_end_models,
        migration_models,
        split_models,
    ) = use_first_order_model(data=data, subsampling=subsampling_factor)
    # constant_new_models, constant_end_models, migration_models, split_models = use_nearest_neighbor(data=data, subsampling=subsampling_factor)
    # constant_new_models, constant_end_models, migration_models, split_models = add_angle_models(data=data, subsampling=subsampling_factor)

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


def load_data(tracking_file: Path, subsampling_factor: int, end_frame: int):
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

    return sub_overlay, sub_tracking_graph


class TestTracking(unittest.TestCase):
    """Test contour functionality"""

    def setUp(self):
        # !wget -O filtered_0.json https://fz-juelich.sciebo.de/s/5vBB6tW8c2DpaU3/download
        # !wget -O 00_stack.tif https://fz-juelich.sciebo.de/s/Xge7fj56QM5ev7q/download

        self.segmentation = Path("filtered_0.json")
        self.image_stack = Path("00_stack.tif")

        if not self.segmentation.exists():
            self.segmentation = wget.download(
                "https://fz-juelich.sciebo.de/s/5vBB6tW8c2DpaU3/download"
            )

        if not self.image_stack.exists():
            self.image_stack = wget.download(
                "https://fz-juelich.sciebo.de/s/Xge7fj56QM5ev7q/download"
            )

    def test_tracking(self):
        tracking_file = self.segmentation

        subsampling_factor = 1
        end_frame = 100

        overlay, _ = load_data(
            tracking_file, subsampling_factor=subsampling_factor, end_frame=end_frame
        )

        output_file = "simpleTracking.json.gz"

        # extract arguments
        num_particles = 1  # args.nb_particles
        num_cores = 1  # args.nb_cpus
        max_num_hypotheses = 1  # args.nb_max_hypotheses
        cutOff = -1  # -10  # args.cutOff
        max_num_solutions = 1  # 10  # args.sol_pool_size

        print("Extract single-cell information...")
        df, all_detections = extract_single_cell_information(overlay)

        print("Setup assignment generators...")
        assignment_generators = setup_assignment_generators(
            df, subsampling_factor=subsampling_factor
        )

        print("Perform tracking...")
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
            mip_method="CBC",  # use CBC as gurobi is not installed in colab (there are gurobi colab examples. Thus, it should be possible to use gurobi)
        )
        end = time.time()

        print("time for tracking", end - start)

        save_tracking(res[0], all_detections, output_file)


if __name__ == "__main__":
    unittest.main()
