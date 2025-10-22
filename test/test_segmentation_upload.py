from pathlib import Path
import pytest
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

    # Set the crate to the secong segmentation to upload to existing image
    uploader.crate = wholecell_segmentation

    # TODO: here we need to set the image ID in the RO-Crate metadata to point to the existing image

    check_seg_dataset(dataset, connection, True, n_rois_expected=2)
