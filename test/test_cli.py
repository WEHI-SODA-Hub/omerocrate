from pathlib import Path
from omero.gateway import BlitzGateway
from typer.testing import CliRunner
from util import check_art_dataset, requires_flower
from omerocrate.cli import app
import pytest

@pytest.mark.parametrize("uploader", [
    "omerocrate.ApiUploader",
    pytest.param("omerocrate.TaskqueueUploader", marks=requires_flower)
])
def test_cli(connection: BlitzGateway, abstract_crate: Path, uploader: str, capsys: pytest.CaptureFixture[str]):
    with capsys.disabled():
        result = CliRunner(mix_stderr=False).invoke(app, [str(abstract_crate), "--uploader-path", uploader])
    assert result.exit_code == 0, f"CLI command failed with error: {result.stderr}"
    # Can't query the dataset unless we are in the right group
    connection.setGroupNameForSession("Abstract art")
    # Parse the dataset ID from the output
    dataset = connection.getObject("Dataset", int(result.stdout))
    check_art_dataset(dataset)
