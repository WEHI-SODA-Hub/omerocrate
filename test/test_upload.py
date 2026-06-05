from pathlib import Path
import pytest
from omerocrate.uploader import ApiUploader, OmeroUploader, OmeroPermissions
from omerocrate.taskqueue.upload import TaskqueueUploader
from omero.gateway import BlitzGateway
from util import check_art_dataset, get_dataset_permissions, requires_flower


@pytest.mark.parametrize(
    "Uploader", [ApiUploader, pytest.param(TaskqueueUploader, marks=requires_flower)]
)
@pytest.mark.asyncio
async def test_upload_api(
    abstract_crate: Path, connection: BlitzGateway, Uploader: type[OmeroUploader]
):
    uploader = Uploader(
        conn=connection, crate=abstract_crate, segmentation_uploader=None
    )
    dataset = await uploader.execute()
    permissions = get_dataset_permissions(dataset)
    assert str(permissions) == "rwra--"
    check_art_dataset(dataset)
    # Test twice to ensure that the tests work with an existing group
    dataset = await uploader.execute()
    check_art_dataset(dataset)


class ReadWriteUploader(ApiUploader):
    def get_group_perms(self) -> OmeroPermissions:
        return OmeroPermissions.ReadWrite


@pytest.mark.asyncio
async def test_upload_readwrite(abstract_crate: Path, connection: BlitzGateway):
    """
    Test that a custom uploader can set the group permissions to ReadWrite
    """
    uploader = ReadWriteUploader(
        conn=connection, crate=abstract_crate, segmentation_uploader=None
    )
    dataset = await uploader.execute()
    check_art_dataset(dataset)
    permissions = get_dataset_permissions(dataset)
    assert str(permissions) == "rwrw--"
