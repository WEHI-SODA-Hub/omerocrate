import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from omero.gateway import (
    BlitzObjectWrapper,
    DatasetWrapper,
    BlitzGateway,
)
from omero.rtypes import rstring

# By importing this module, certain fields are populated on the omero module
import omero.api  # pyright: ignore[reportUnusedImport]
import omero
import pytest
from omerocrate.utils import delete_dataset
from dotenv import get_key
from omero_model_PermissionsI import PermissionsI
from omero_model_ExperimenterGroupI import ExperimenterGroupI
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from omero_api_IAdmin_ice import IAdminPrx


def check_art_dataset(dataset: DatasetWrapper):
    """
    Check if the test dataset has been uploaded correctly
    """
    assert dataset.name == "Abstract art"
    assert dataset.countChildren() == 1
    assert dataset.getDetails().getGroup().getName() == "Abstract art", (
        "The dataset group should be the crate name"
    )
    for image in dataset.listChildren():
        assert "Color Study" in image.name
    delete_dataset(dataset)


def get_dataset_permissions(wrapper: BlitzObjectWrapper) -> PermissionsI:
    return wrapper.getDetails().getGroup().getDetails().getPermissions()


def archive_group(group: ExperimenterGroupI, connection: BlitzGateway) -> None:
    """
    Removes all members from a group and renames it to ``archived_<uuid>`` so
    that the original group name can be reused without collision.
    """
    admin: IAdminPrx = connection.getAdminService()

    for experimenter in group.linkedExperimenterList():
        admin.removeGroups(experimenter, [group])

    group.name = rstring(f"archived_{uuid.uuid4().hex}")
    admin.updateGroup(group)


@contextmanager
def using_group(
    group_name: str, connection: BlitzGateway
) -> Generator[None, None, None]:
    """
    Context manager that ensures ``group_name`` is free before and after use.

    On entry, any existing groups with ``group_name`` are archived.
    On exit, any group that was created during the body with ``group_name`` is
    archived.
    """
    admin: IAdminPrx = connection.getAdminService()

    try:
        existing = admin.lookupGroup(group_name)
        archive_group(existing, connection)
    except omero.ApiUsageException:
        # Group doesn't exist, which is what we want
        pass

    yield

    try:
        existing = admin.lookupGroup(group_name)
        archive_group(existing, connection)
    except omero.ApiUsageException:
        pass


def check_seg_dataset(
    dataset: DatasetWrapper,
    conn: BlitzGateway,
    check_rois: bool = False,
    n_rois_expected: int = 0,
):
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
requires_flower = pytest.mark.skipif(
    not (os.environ.get("FLOWER_HOST") or get_key(root / ".env", "FLOWER_HOST")),
    reason="OMERO taskqueue not available",
)
requires_roi_tool = pytest.mark.skipif(
    os.getenv("RUNNER_ENVIRONMENT") == "github-hosted",
    reason="ROI tool not available on GitHub-hosted runners",
)
