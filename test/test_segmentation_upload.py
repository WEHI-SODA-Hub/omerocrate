from pathlib import Path
import pytest
from omerocrate.uploader import OmeroUploader, SegmentationUploader
from omerocrate.taskqueue.upload import TaskqueueUploader
from omero.gateway import BlitzGateway
from util import requires_flower


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
    #check_art_dataset(dataset)
