import pytest
from pathlib import Path
from omerocrate.utils import uri_to_path


def test_uri_to_path_valid_file_uri():
    uri = "file:///home/user/data/image.tif"
    result = uri_to_path(uri)
    assert result == Path("/home/user/data/image.tif")


def test_uri_to_path_percent_encoded():
    uri = "file:///home/user/my%20data/image%2B1.tif"
    result = uri_to_path(uri)
    assert result == Path("/home/user/my data/image+1.tif")


def test_uri_to_path_invalid_scheme_raises():
    uri = "https://example.com/image.tif"
    with pytest.raises(ValueError, match="URI scheme must be 'file'"):
        uri_to_path(uri)
