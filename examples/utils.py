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

    with VideoExporter(output_file, framerate, codec) as ve:
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
