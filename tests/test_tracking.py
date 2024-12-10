""" Test tracking models
"""
import time
import unittest
from pathlib import Path

import wget

from uatrack.config import setup_assignment_generators
from uatrack.core import simpleTracking
from uatrack.utils import extract_single_cell_information, load_data, save_tracking


class TestTracking(unittest.TestCase):
    """Test contour functionality"""

    def setUp(self):
        self.segmentation = Path("filtered_0.json")
        self.image_stack = Path("00_stack.tif")

        # Download data if needed
        if not self.segmentation.exists():
            self.segmentation = wget.download(
                "https://fz-juelich.sciebo.de/s/5vBB6tW8c2DpaU3/download"
            )

        if not self.image_stack.exists():
            self.image_stack = wget.download(
                "https://fz-juelich.sciebo.de/s/Xge7fj56QM5ev7q/download"
            )

    def tracking(self, model="NN", end_frame=100):
        tracking_file = self.segmentation

        subsampling_factor = 1

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
            df, subsampling_factor=subsampling_factor, model=model
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

    def test__nn_tracking(self):
        self.tracking(model="NN")

    def test__fo_tracking(self):
        self.tracking(model="FO")

    def test__fo_g_tracking(self):
        self.tracking(model="FO+G")

    def test__fo_o_tracking(self):
        self.tracking(model="FO+O")

    def test__fo_dd_tracking(self):
        self.tracking(model="FO+DD")

    def test__fo_g_o_dd_tracking(self):
        self.tracking(model="FO+G+O+DD")


if __name__ == "__main__":
    unittest.main()
