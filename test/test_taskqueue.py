from omerocrate.taskqueue.models import UploadRequest
from pydantic import TypeAdapter
from pathlib import Path

def test_parse_model():
    examples = Path(__file__).parent / "examples.json"
    results = TypeAdapter(list[UploadRequest]).validate_json(examples.read_text())
    assert len(results) == 2
