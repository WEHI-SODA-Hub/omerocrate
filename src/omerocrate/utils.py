from pathlib import Path
from omero.gateway import (
    DatasetWrapper,
    ExperimenterGroupWrapper,
    ExperimenterWrapper,
    ProxyObjectWrapper,
)
from urllib.parse import urlsplit, unquote


def delete_dataset(dataset: DatasetWrapper):
    """
    Delete a dataset and all its children.
    """
    dataset._conn.deleteObjects(
        "Dataset",
        [dataset._obj.id.val],
        deleteChildren=True,
        deleteAnns=True,
        wait=True,
    )


def user_in_group(
    user: ExperimenterWrapper,
    group: ExperimenterGroupWrapper,
    admin_service: ProxyObjectWrapper,
) -> bool:
    """
    Check if a user is a member of a group.
    """
    return user.getId() in {
        user.id.val for user in admin_service.containedExperimenters(group.getId())
    }


def uri_to_path(uri: str) -> Path:
    """
    Convert a URI to a file path, checking that the scheme is 'file', and unquoting any percent-encoded characters.
    """
    parsed = urlsplit(uri)
    if parsed.scheme != "file":
        raise ValueError(f"URI scheme must be 'file', got '{parsed.scheme}'")
    return Path(unquote(parsed.path))
