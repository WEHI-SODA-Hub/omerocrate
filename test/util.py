import os
from pathlib import Path
from omero.gateway import DatasetWrapper
import pytest
from omerocrate.utils import delete_dataset
from omero.gateway import BlitzGateway
from dotenv import get_key

def check_art_dataset(dataset: DatasetWrapper):
    """
    Check if the test dataset has been uploaded correctly
    """
    assert dataset.name == "Abstract art"
    assert dataset.countChildren() == 1
    assert dataset.getDetails().getGroup().getName() == "Abstract art", "The dataset group should be the crate name"
    for image in dataset.listChildren():
        assert "Color Study" in image.name
    delete_dataset(dataset)


def check_seg_dataset(dataset: DatasetWrapper, conn: BlitzGateway, check_rois: bool = False,
                      n_rois_expected: int = 0):
    """
    Check if the test segmentation dataset has been uploaded correctly
    """
    assert dataset.name == "Nuclear image"
    assert dataset.countChildren() == 1
    roi_service = conn.getRoiService()
    for image in dataset.listChildren():
        assert "Nuclear image" in image.name
        if check_rois:
            result = roi_service.findByImage(image.getId(), None)
            assert len(result.rois) == n_rois_expected, "No ROIs found for image"
    roi_service.close()
    delete_dataset(dataset)


root = Path(__file__).parent.parent
requires_flower= pytest.mark.skipif(not (os.environ.get("FLOWER_HOST") or get_key(root / ".env", "FLOWER_HOST")), reason="OMERO taskqueue not available")
requires_roi_tool = pytest.mark.skipif(os.getenv("RUNNER_ENVIRONMENT") == "github-hosted", reason="ROI tool not available on GitHub-hosted runners")
