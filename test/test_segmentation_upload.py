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
    check_roi = SegUploader is not None
    check_seg_dataset(dataset, connection, check_roi)
