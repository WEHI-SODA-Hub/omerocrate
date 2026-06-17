from omerocrate.taskqueue.models import UploadRequest, UploadResultSet
from pydantic import TypeAdapter
from pathlib import Path

root = Path(__file__).parent / "taskqueue_examples"


def test_request_model():
    examples = root / "upload_request.json"
    results = TypeAdapter(list[UploadRequest]).validate_json(examples.read_text())
    assert len(results) == 2


def test_response_model():
    examples = root / "result.json"
    results = TypeAdapter(list[UploadResultSet]).validate_json(examples.read_text())
    assert len(results) == 1
