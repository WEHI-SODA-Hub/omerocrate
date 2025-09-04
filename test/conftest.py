from pathlib import Path
from git import Repo
import pytest
from omerocrate.gateway import from_env
from omero.gateway import BlitzGateway
import dotenv
import urllib.request

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
