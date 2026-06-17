from omerocrate.taskqueue.models import ImportSummary, UploadRequest, UploadResultSet
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
    img = results[0].result[0].project[0].dataset[0].image[0]
    assert img.object_id == [747, 748, 749]
    assert isinstance(img.import_summary, ImportSummary)
    assert img.import_summary.image_id == [747, 748, 749]
    assert len(results) == 1
