from pathlib import Path
from unittest.mock import MagicMock

import omero
import pytest
from rdflib import Graph, Literal, Namespace, URIRef

from omerocrate.uploader import OmeroUploader

SCHEMA = Namespace("http://schema.org/")


def image_uri(tmp_path: Path, filename: str) -> URIRef:
    return URIRef((tmp_path / filename).as_uri())


@pytest.fixture
def uploader(tmp_path: Path) -> OmeroUploader:
    """OmeroUploader with an empty in-memory graph, no real OMERO connection."""
    uploader = OmeroUploader.model_construct(conn=MagicMock(), crate=tmp_path)
    # Patch the graph to be an empty in-memory graph instead of loading from crate metadata
    uploader.graph = Graph()
    return uploader


@pytest.fixture(autouse=True)
def instant_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run tests instantly."""
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)


@pytest.fixture
def stale_image() -> MagicMock:
    """Image wrapper as yielded by upload_images, potentially stale."""
    image = MagicMock()
    image.getId.return_value = 42
    return image


class TestProcessImage:
    def test_saves_onto_refetched_copy(
        self, uploader: OmeroUploader, tmp_path: Path, stale_image: MagicMock
    ):
        """
        Metadata should be applied to a freshly fetched image and saved there,
        never on the (potentially stale) wrapper passed in.
        """
        uri = image_uri(tmp_path, "image.tif")
        uploader.graph.add((uri, SCHEMA.name, Literal("Curated name")))
        uploader.graph.add((uri, SCHEMA.description, Literal("Curated description")))
        fresh_image = uploader.conn.getObject.return_value

        uploader.process_image(uri, stale_image)

        uploader.conn.getObject.assert_called_once_with("Image", 42)
        fresh_image.setName.assert_called_once_with("Curated name")
        fresh_image.setDescription.assert_called_once_with("Curated description")
        fresh_image.save.assert_called_once()
        stale_image.save.assert_not_called()

    def test_retries_on_optimistic_lock_exception(
        self, uploader: OmeroUploader, tmp_path: Path, stale_image: MagicMock
    ):
        """
        If saving raises OptimisticLockException, the image should be refetched
        and the save retried until it succeeds.
        """
        uri = image_uri(tmp_path, "image.tif")
        uploader.graph.add((uri, SCHEMA.name, Literal("Curated name")))
        fresh = uploader.conn.getObject.return_value
        fresh.save.side_effect = [omero.OptimisticLockException(), None]

        uploader.process_image(uri, stale_image)

        assert uploader.conn.getObject.call_count == 2
        assert fresh.save.call_count == 2

    def test_gives_up_after_three_attempts(
        self, uploader: OmeroUploader, tmp_path: Path, stale_image: MagicMock
    ):
        """
        If every save attempt conflicts, the exception should propagate after
        three attempts.
        """
        uri = image_uri(tmp_path, "image.tif")
        uploader.graph.add((uri, SCHEMA.name, Literal("Curated name")))
        fresh = uploader.conn.getObject.return_value
        fresh.save.side_effect = omero.OptimisticLockException()

        with pytest.raises(omero.OptimisticLockException):
            uploader.process_image(uri, stale_image)

        assert uploader.conn.getObject.call_count == 3
        assert fresh.save.call_count == 3

    def test_no_metadata_skips_save(
        self, uploader: OmeroUploader, tmp_path: Path, stale_image: MagicMock
    ):
        """
        With no name or description in the crate, nothing should be fetched or
        saved at all.
        """
        uri = image_uri(tmp_path, "image.tif")

        uploader.process_image(uri, stale_image)

        uploader.conn.getObject.assert_not_called()
        stale_image.save.assert_not_called()
