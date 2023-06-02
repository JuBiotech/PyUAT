""" Example for performing the tracking analysis in parallel for multiple sequences and uploading the results"""
from pathlib import Path

from acia.segm.omero.utils import list_image_ids_in
from big_example import main
from credentials import host, password, port, username
from omero.gateway import BlitzGateway

# OMERO resource that you want to analyze
omero_type = "dataset"  # can be "image", "project" or "dataset"
omero_id = 2351  # change the id if you want to apply the analysis to a different omero resource

credentials = dict(
    serverUrl=host,
    username=username,
    password=password,
    port=port,
)

omero_cred = dict(
    host=credentials["serverUrl"],
    username=credentials["username"],
    passwd=credentials["password"],
    port=credentials["port"],
    secure=True,
)

with BlitzGateway(**omero_cred) as conn:
    image_ids = list_image_ids_in(omero_id, omero_type, conn)

## TODO: give an overview about the data
print(image_ids)

for image_id in image_ids:
    output_folder = Path("tracking_output") / f"{image_id}"

    main(omero_id=image_id, output_folder=output_folder)
