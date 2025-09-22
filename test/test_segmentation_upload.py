from pathlib import Path
import pytest
import os
from omerocrate.uploader import OmeroUploader, SegmentationUploader
from omerocrate.taskqueue.upload import TaskqueueUploader
from omero.gateway import BlitzGateway
from util import requires_flower, check_seg_dataset


@pytest.mark.skipif(
    os.getenv("RUNNER_ENVIRONMENT") == "github-hosted",
    reason="Skip on GitHub-hosted runners due to ROI tool requirement, only run on self-hosted"
)
@pytest.mark.parametrize("Uploader", [
    SegmentationUploader,
    pytest.param(TaskqueueUploader, marks=requires_flower)
])
@pytest.mark.asyncio
async def test_segmentation_upload(nuclear_image: Path, connection: BlitzGateway,
                                   Uploader: type[OmeroUploader]):
    uploader = Uploader(
        conn=connection,
        crate=nuclear_image
    )
    dataset = await uploader.execute()
    check_seg_dataset(dataset)
