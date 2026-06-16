from omero.gateway import BlitzGateway
import os
from getpass import getuser

def from_env() -> BlitzGateway:
    """
    Authenticate to OMERO using environment variables.
    You will need to set the following environment variables:

    - `OMERO_PASSWORD`
    - `OMERO_HOST`
    - `OMERO_USER` (optional, defaults to the current username)
    - `OMERO_PORT` (optional, defaults to 4064)
    - `OMERO_SECURE` (optional, defaults to True)
    """
    return BlitzGateway(
        username=os.environ.get("OMERO_USER", getuser()),
        passwd=os.environ["OMERO_PASSWORD"],
        host=os.environ["OMERO_HOST"],
        port=os.environ.get("OMERO_PORT", 4064),
        secure=bool(os.environ.get("OMERO_SECURE", "1").lower() in ("1", "true", "yes")),
    )
