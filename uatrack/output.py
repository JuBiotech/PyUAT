""" Module for tracking result output generation """

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from pandas import DataFrame

from uatrack.core import SimpleCluster


class SimpleTrackingReporter:
    """Generate output in the simple tracking format"""

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

        print(self.output_folder)

        with gzip.open(
            Path(self.output_folder) / "tracking.json.gz", "wt"
        ) as output_file:
            json.dump(data_structure, output_file)
        # print(self.final_cluster.tracking.createIndexTracking().edges)
