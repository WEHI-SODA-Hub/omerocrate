from pathlib import Path
import json
import shutil
import tempfile
import pytest
from omerocrate.uploader import ApiUploader, OmeroUploader, OmeroPermissions
from omerocrate.taskqueue.upload import TaskqueueUploader
from omero.gateway import BlitzGateway
from util import check_art_dataset, get_dataset_permissions, requires_flower
from omerocrate.utils import delete_dataset


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

    def get_group_name(self) -> str:
        return "Abstract art (ReadWrite)"


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


@pytest.mark.asyncio
async def test_upload_two_image_names(abstract_crate: Path, connection: BlitzGateway):
    """
    Test that the uploader succeeds when the image node has two schema:name predicates.
    process_image() uses select_first(), so it should pick one name without raising.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_crate = Path(tmp_dir) / "crate"
        shutil.copytree(abstract_crate, temp_crate)

        metadata_path = temp_crate / "ro-crate-metadata.json"
        with open(metadata_path) as f:
            crate_data = json.load(f)

        for item in crate_data["@graph"]:
            if item.get("@id") == "concentric.jpg":
                item["name"] = [
                    "Color Study. First Name",
                    "Color Study. Second Name",
                ]
                break

        with open(metadata_path, "w") as f:
            json.dump(crate_data, f)

        uploader = ApiUploader(conn=connection, crate=temp_crate)
        dataset = await uploader.execute()

        assert dataset.countChildren() == 1
        for image in dataset.listChildren():
            assert image.name in (
                "Color Study. First Name",
                "Color Study. Second Name",
            )
        delete_dataset(dataset)
