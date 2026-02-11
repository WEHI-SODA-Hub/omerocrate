from pathlib import Path
import pytest
from omerocrate.uploader import ApiUploader, OmeroUploader
from omerocrate.taskqueue.upload import TaskqueueUploader
from omero.gateway import BlitzGateway
from util import check_art_dataset, requires_flower


@pytest.mark.parametrize("Uploader", [
    ApiUploader,
    pytest.param(TaskqueueUploader, marks=requires_flower)
])
@pytest.mark.asyncio
async def test_upload_api(abstract_crate: Path, connection: BlitzGateway,
                          Uploader: type[OmeroUploader]):
    uploader = Uploader(
        conn=connection,
        crate=abstract_crate,
        segmentation_uploader=None
    )
    dataset = await uploader.execute()
    check_art_dataset(dataset)
    # Test twice to ensure that the tests work with an existing group
    dataset = await uploader.execute()
    check_art_dataset(dataset)

@pytest.mark.parametrize("Uploader", [
    ApiUploader,
    pytest.param(TaskqueueUploader, marks=requires_flower)
])
@pytest.mark.asyncio
async def test_upload_empty(connection: BlitzGateway, Uploader: type[OmeroUploader], caplog: pytest.LogCaptureFixture):
    """
    Check that uploading a crate with metadata but without any files marked for upload still works.
    """
    uploader = Uploader(
        conn=connection,
        crate=Path(__file__).parent / "empty_crate",
        segmentation_uploader=None
    )
    dataset = await uploader.execute()
    # Check that the dataset has been created correctly
    check_art_dataset(dataset)
    assert dataset.countChildren() == 0, "Dataset should have no images"
    assert "No files" in caplog.text
