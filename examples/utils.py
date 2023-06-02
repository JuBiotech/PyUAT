""" Utilities for the tracking """

import cv2
import numpy as np
from acia.base import Overlay
from acia.segm.output import VideoExporter
from networkx import DiGraph
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from uat.core import SimpleCluster


def render_tracking_video(
    image_source,
    full_overlay,
    tracking,
    output_file="track.avi",
    framerate=3,
    codec="MJPG",
):

    all_contours = list(full_overlay)
    contour_lookup = {cont.id: cont for cont in all_contours}

    with VideoExporter(str(output_file), framerate, codec) as ve:
        for frame, (image, overlay) in enumerate(
            tqdm(zip(image_source, full_overlay.timeIterator()))
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


def cluster_to_tracking_graph(cluster: SimpleCluster, overlay: Overlay):

    detections = overlay.contours

    index_edge_list = list(
        map(
            lambda e: (int(e[0]), int(e[1])),
            cluster.tracking.createIndexTracking().edges,
        )
    )

    id_edge_list = [
        (detections[edge[0]].id, detections[edge[1]].id) for edge in index_edge_list
    ]

    g = DiGraph()
    g.add_nodes_from([d.id for d in detections])
    g.add_edges_from(id_edge_list)

    return g


def compute_errors(sol_tracking_graph: DiGraph, pred_tracking_graph: DiGraph):
    """Compares a solution tracking graph with a predicted tracking graph and prints some error measures

    Args:
        sol_tracking_graph (DiGraph): solution tracking graph (with detection ids as nodes)
        pred_tracking_graph (DiGraph): predicted tracking graph (with detection ids as nodes)
    """

    # first only use the solution nodes also present in the prediction
    sol_tracking_graph = sol_tracking_graph.subgraph(pred_tracking_graph.nodes)

    # choose short names
    stg = sol_tracking_graph
    ptg = pred_tracking_graph

    # make sure that the node sets are the same
    assert (
        stg.number_of_nodes() == ptg.number_of_nodes()
    ), f"You are comparing tracking solution and prediction with different detection sets! sol nodes={stg.number_of_nodes()} vs. pred nodes={ptg.number_of_nodes()}"

    # get edge sets
    sol_edge_set = set(stg.edges)
    pred_edge_set = set(ptg.edges)

    # Compute metrics

    intersect = sol_edge_set.intersection(pred_edge_set)

    print(f"Num edges in solution: {len(sol_edge_set)}")
    print(f"Num edges in prediction: {len(pred_edge_set)}")
    print(f"Num edges in intersect: {len(intersect)}")

    TP = len(intersect)
    FP = len(pred_edge_set - sol_edge_set)
    FN = len(sol_edge_set - pred_edge_set)

    print(f"Num TP: {TP}")
    print(f"Num FP: {FP}")
    print(f"Num FN: {FN}")

    print(f"False positives: {pred_edge_set - sol_edge_set}")
