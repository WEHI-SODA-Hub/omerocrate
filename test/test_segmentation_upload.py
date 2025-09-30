from pathlib import Path
import pytest
import os
from omerocrate.uploader import OmeroUploader, ApiUploader, SegmentationUploader, OmeNgffUploader
from omerocrate.taskqueue.upload import TaskqueueUploader
from omero.gateway import BlitzGateway
from util import requires_flower, check_seg_dataset


@pytest.mark.skipif(
    os.getenv("RUNNER_ENVIRONMENT") == "github-hosted",
    reason="Skip on GitHub-hosted runners due to ROI tool requirement, only run on self-hosted"
)
@pytest.mark.parametrize("Uploader", [
    ApiUploader,
    pytest.param(TaskqueueUploader, marks=requires_flower)
])
@pytest.mark.parametrize("SegUploader", [
    OmeNgffUploader
])
@pytest.mark.asyncio
async def test_segmentation_upload(nuclear_image: Path, connection: BlitzGateway,
                                   Uploader: type[OmeroUploader],
                                   SegUploader: type[SegmentationUploader]):
    uploader = Uploader(
        conn=connection,
        crate=nuclear_image,
        segmentation_uploader=SegUploader(conn=connection, upload_directory=None),
    )
    dataset = await uploader.execute()
    check_seg_dataset(dataset, connection)
