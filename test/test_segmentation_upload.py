from pathlib import Path
import pytest
import json
import tempfile
import shutil
from omerocrate.uploader import OmeroUploader, ApiUploader, SegmentationUploader, OmeNgffUploader
from omerocrate.taskqueue.upload import TaskqueueUploader
from omero.gateway import BlitzGateway
from util import requires_flower, requires_roi_tool, check_seg_dataset


@pytest.mark.parametrize("Uploader", [
    ApiUploader,
    pytest.param(TaskqueueUploader, marks=requires_flower)
])
@pytest.mark.parametrize("SegUploader", [
    None,
    pytest.param(OmeNgffUploader, marks=requires_roi_tool),
])
@pytest.mark.asyncio
async def test_segmentation_upload(nuclear_image: Path, connection: BlitzGateway,
                                   Uploader: type[OmeroUploader],
                                   SegUploader: type[SegmentationUploader]):
    seg_uploader = SegUploader(conn=connection, upload_directory=None) if SegUploader else None
    uploader = Uploader(
        conn=connection,
        crate=nuclear_image,
        segmentation_uploader=seg_uploader,
    )
    dataset = await uploader.execute()
    check_rois = SegUploader is not None
    check_seg_dataset(dataset, connection, check_rois, n_rois_expected=1)


@pytest.mark.parametrize("Uploader", [
    ApiUploader,
    pytest.param(TaskqueueUploader, marks=requires_flower)
])
@pytest.mark.parametrize("SegUploader", [
    pytest.param(OmeNgffUploader, marks=requires_roi_tool),
])
@pytest.mark.asyncio
async def test_segmentation_upload_existing_image(nuclear_image: Path,
                                                  wholecell_segmentation: Path,
                                                  connection: BlitzGateway,
                                                  Uploader: type[OmeroUploader],
                                                  SegUploader: type[SegmentationUploader]):
    seg_uploader = SegUploader(conn=connection, upload_directory=None) if SegUploader else None

    # Upload the image first
    uploader = Uploader(
        conn=connection,
        crate=nuclear_image,
        segmentation_uploader=seg_uploader,
    )
    dataset = await uploader.execute()

    # Get image ID for setting in the RO-Crate
    image_id = None
    for image in dataset.listChildren():
        image_id = image.getId()

    # We need to modify the RO-Crate to set the image ID to the existing image
    # Create a temporary copy of the crate to modify
    temp_crate_dir = tempfile.mkdtemp()
    temp_crate_path = Path(temp_crate_dir)
    shutil.copytree(wholecell_segmentation, temp_crate_path / "crate", dirs_exist_ok=True)

    metadata_path = temp_crate_path / "crate" / "ro-crate-metadata.json"
    with open(metadata_path, 'r') as f:
        crate_data = json.load(f)

    for item in crate_data["@graph"]:
        if item.get("@id") == "nuclear_image.tif":
            item["imageID"] = str(image_id)
            break

    with open(metadata_path, 'w') as f:
        json.dump(crate_data, f, indent=4)

    # Make a new uploader with the temporary crate
    uploader = Uploader(
        conn=connection,
        crate=temp_crate_path / "crate",
        segmentation_uploader=seg_uploader,
    )
    dataset = await uploader.execute()

    check_seg_dataset(dataset, connection, True, n_rois_expected=2)
