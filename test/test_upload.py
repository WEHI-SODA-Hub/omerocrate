from pathlib import Path
import json
import shutil
import tempfile
import pytest
from rdflib import Graph, Literal, URIRef
from omerocrate.uploader import ApiUploader, OmeroUploader, OmeroPermissions
from omerocrate.taskqueue.upload import TaskqueueUploader
from omero.gateway import BlitzGateway
from omerocrate.utils import delete_dataset
from util import (
    check_art_dataset,
    get_dataset_permissions,
    requires_flower,
    using_group,
)


@pytest.mark.parametrize(
    "Uploader", [ApiUploader, pytest.param(TaskqueueUploader, marks=requires_flower)]
)
@pytest.mark.asyncio
async def test_upload_api(
    abstract_crate: Path,
    connection: BlitzGateway,
    Uploader: type[OmeroUploader],
):
    with using_group("Abstract art", connection):
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


@pytest.mark.parametrize(
    "Uploader", [ApiUploader, pytest.param(TaskqueueUploader, marks=requires_flower)]
)
@pytest.mark.asyncio
async def test_upload_multi_image(
    multi_image_abstract_crate: Path,
    connection: BlitzGateway,
    Uploader: type[OmeroUploader],
):
    with using_group("Multi Image Abstract Art", connection):
        uploader = Uploader(
            conn=connection, crate=multi_image_abstract_crate, segmentation_uploader=None
        )
        dataset = await uploader.execute()
        permissions = get_dataset_permissions(dataset)
        assert str(permissions) == "rwra--"
        assert dataset.name == "Multi Image Abstract Art"
        assert dataset.countChildren() == 2
        assert (
            dataset.getDetails().getGroup().getName() == "Multi Image Abstract Art"
        ), "The dataset group should be the crate name"
        assert {image.name for image in dataset.listChildren()} == {
            "Color Study. Squares with Concentric Circles",
            "Accent on rose",
        }
        delete_dataset(dataset)


class ReadWriteUploader(ApiUploader):
    def get_group_perms(self) -> OmeroPermissions:
        return OmeroPermissions.ReadWrite


@pytest.mark.asyncio
async def test_upload_readwrite(abstract_crate: Path, connection: BlitzGateway):
    """
    Test that a custom uploader can set the group permissions to ReadWrite
    """
    with using_group("Abstract art", connection):
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
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        using_group("Abstract art", connection),
    ):
        temp_crate = Path(tmp_dir) / "crate"
        shutil.copytree(abstract_crate, temp_crate)

        metadata_path = temp_crate / "ro-crate-metadata.json"
        graph = Graph().parse(metadata_path.as_posix(), format="json-ld")

        # Add the second name for the image
        graph.add(
            (
                URIRef("concentric.jpg"),
                URIRef("http://schema.org/name"),
                Literal("Second Name"),
            )
        )

        with open(metadata_path, "wb") as f:
            graph.serialize(f, format="json-ld")

        uploader = ApiUploader(conn=connection, crate=temp_crate)
        dataset = await uploader.execute()

        assert dataset.countChildren() == 1
        for image in dataset.listChildren():
            # We don't care which name is picked, just that it doesn't raise an error and that the name is one of the two we set
            assert image.name in (
                "Color Study. Squares with Concentric Circles",
                "Second Name",
            )
        delete_dataset(dataset)
