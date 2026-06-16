from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rdflib import Graph, Literal, Namespace, URIRef

from omerocrate.uploader import OmeroUploader

OME = Namespace("http://www.openmicroscopy.org/Schemas/OME/2016-06/")


def image_uri(tmp_path: Path, filename: str) -> URIRef:
    return URIRef((tmp_path / filename).as_uri())


@pytest.fixture
def uploader(tmp_path: Path) -> OmeroUploader:
    """OmeroUploader with an empty in-memory graph, no real OMERO connection."""
    uploader = OmeroUploader.model_construct(conn=MagicMock(), crate=tmp_path)
    # Patch the graph to be an empty in-memory graph instead of loading from crate metadata
    uploader.graph = Graph()
    return uploader


class TestFindExistingImages:
    def test_empty_list_returns_nothing(self, uploader: OmeroUploader, tmp_path: Path):
        """
        With images in the crate but no URIs provided, nothing should be returned.
        """
        uploader.graph.add((image_uri(tmp_path, "image.tif"), OME.ImageID, Literal(1)))
        uploader.graph.serialize(
            destination=tmp_path / "ro-crate-metadata.json"
        )  # Ensure graph is saved to crate
        result = list(uploader.find_existing_images([]))
        assert result == []

    def test_single_image_found(self, uploader: OmeroUploader, tmp_path: Path):
        """
        With one image in the crate which has an OMERO ID, and a matching URI provided, that image and its OMERO ID should be returned.
        """
        uploader.graph.add(
            (uri := image_uri(tmp_path, "image.tif"), OME.ImageID, Literal(42))
        )
        result = list(uploader.find_existing_images([uri]))
        assert result == [(uri, 42)]

    def test_single_image_not_in_graph(self, uploader: OmeroUploader, tmp_path: Path):
        """
        With no images in the crate, and a URI provided, nothing should be returned.
        """
        uri = image_uri(tmp_path, "missing.tif")
        result = list(uploader.find_existing_images([uri]))
        assert result == []

    def test_multiple_images_all_found(self, uploader: OmeroUploader, tmp_path: Path):
        """
        With multiple images in the crate which have OMERO IDs, and matching URIs provided, all images and their OMERO IDs should be returned.
        """
        uploader.graph.add(
            (img1 := image_uri(tmp_path, "img1.tif"), OME.ImageID, Literal(10))
        )
        uploader.graph.add(
            (img2 := image_uri(tmp_path, "img2.tif"), OME.ImageID, Literal(20))
        )
        uploader.graph.add(
            (img3 := image_uri(tmp_path, "img3.tif"), OME.ImageID, Literal(30))
        )
        result = dict(uploader.find_existing_images([img1, img2, img3]))

        assert result == {img1: 10, img2: 20, img3: 30}

    def test_multiple_images_partial_match(
        self, uploader: OmeroUploader, tmp_path: Path
    ):
        """
        With one image in the crate which has an OMERO ID, and two URIs provided where only one matches, only the matching image and its OMERO ID should be returned.
        """
        absent_uri = image_uri(tmp_path, "absent.tif")
        uploader.graph.add(
            (
                present_uri := image_uri(tmp_path, "present.tif"),
                OME.ImageID,
                Literal(99),
            )
        )

        result = dict(uploader.find_existing_images([present_uri, absent_uri]))

        assert result == {present_uri: 99}
