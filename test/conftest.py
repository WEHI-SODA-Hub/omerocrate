from __future__ import annotations
from pathlib import Path
from git import Repo
import pytest
from omerocrate.gateway import from_env
from omero.gateway import (
    BlitzGateway,
    ExperimenterGroupWrapper,
)
from typing import TYPE_CHECKING
import dotenv
import urllib.request

if TYPE_CHECKING:
    from omero_api_IAdmin_ice import IAdminPrx


@pytest.fixture
def abstract_crate() -> Path:
    return Path(__file__).parent / "demo_crate"


@pytest.fixture
def ca_imaging() -> Path:
    out = Path(__file__).parent / "ca-imaging"
    if not out.exists():
        Repo.clone_from("https://github.com/SFB-ELAINE/Ca-imaging-RO-Crate", out)
    return out


@pytest.fixture
def ca_imaging_1021(ca_imaging: Path) -> Path:
    return ca_imaging / "ro-crate_1021"


@pytest.fixture
def nuclear_image() -> Path:
    out = Path(__file__).parent / "demo_segmentation"
    if not (out / "nuclear_image.tif").exists():
        url = "https://github.com/nf-core/test-datasets/raw/refs/heads/modules/data/imaging/segmentation/nuclear_image.tif"
        urllib.request.urlretrieve(url, out / "nuclear_image.tif")
    return out


@pytest.fixture
def wholecell_segmentation() -> Path:
    return Path(__file__).parent / "demo_segmentation_with_imageid"


@pytest.fixture
def connection() -> BlitzGateway:
    # To run the tests, each user will need to provide credentials for their own OMERO server
    # .env is a convenient way to store these credentials
    dotenv.load_dotenv()
    conn = from_env()
    conn.connect()
    return conn


@pytest.fixture(autouse=True)
def load_env():
    """
    Load environment variables from .env file for testing.
    """
    dotenv.load_dotenv()


@pytest.fixture
def clean_groups(connection: BlitzGateway):
    """
    Deletes any OMERO experimenter groups created during a test.
    Prevents group name collisions between test runs.
    """
    from omero import model as omero_model

    existing_ids = {g.getId() for g in connection.listGroups()}
    yield
    admin: IAdminPrx = connection.getAdminService()
    group: ExperimenterGroupWrapper
    for group in connection.listGroups():
        if group.getId() not in existing_ids:
            admin.deleteGroup(omero_model.ExperimenterGroupI(group.getId()))
