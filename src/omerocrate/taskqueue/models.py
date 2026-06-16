"""
Models for parsing and serializing data structures used in `gs-taskqueue`
"""
from __future__ import annotations
from datetime import datetime
from typing import List, Any, Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict, AliasChoices

def parse_object_id(value: str | int | list | None) -> int | None:
    """
    Parse an OMERO object ID
    """
    if isinstance(value, list):
        if len(value) > 1:
            raise ValueError(f"Invalid object ID: {value}")
        elif len(value) == 1:
            return parse_object_id(value[0])
        else:
            return None
    
    elif value is None:
        return None

    else:
        return int(value)


KeyValue = tuple[str, Any]
Error = Annotated[Optional[str], Field(
    description="Error message if the task failed",
    default=None
)]
OmeroId = Annotated[Optional[int], Field(
    description="Some OMERO ID"
), BeforeValidator(parse_object_id)]
#: An OMERO ID whose field name is objectId
ObjectId = Annotated[OmeroId, Field(
    validation_alias=AliasChoices("objectId", "object_id"),
    alias="objectId"
)]

class TaskQueueBase(BaseModel):
    """
    Base class for all models in this module
    """
    model_config = ConfigDict(
        serialize_by_alias=True,
    )

class ImageRequest(TaskQueueBase):
    """
    Fields needed to upload an image to OMERO
    """
    name: Annotated[str, Field(description="Image name")]
    description: Annotated[Optional[str], Field(description="Image description")] = None
    file_path: Annotated[str, Field(
        description="Path to the image file on disk",
        validation_alias=AliasChoices("filePath", "file_path"),
        alias="filePath"
    )]
    key_values: list[KeyValue] = Field(
        default_factory=list,
        description="List of key-value pairs associated with this image",
        alias="keyValues"
    )
    tag: list[KeyValue] = Field(
        default_factory=list,
        description="List of tags associated with this image"
    )

class DatasetFields(TaskQueueBase):
    """
    Common fields for Dataset and DatasetRequest
    """
    name: Annotated[Optional[str], Field(description="Dataset name")] = None
    object_id: ObjectId = None
    description: Annotated[Optional[str], Field(description="Dataset description")] = None
    key_values: list[KeyValue] = Field(
        default_factory=list,
        description="List of key-value pairs to be added to this dataset",
        alias="keyValues",
        validation_alias=AliasChoices("keyValues", "key_values")
    )
    tag: list[KeyValue] = Field(
        default_factory=list,
        description="List of tags associated with this dataset"
    )


class DatasetRequest(DatasetFields):
    """
    Model for creating a dataset
    """
    image: list[ImageRequest] = Field(
        default_factory=list,
        description="List of images in this dataset"
    )

class ProjectFields(TaskQueueBase):
    """
    Common fields relating to an OMERO Project
    """
    name: Annotated[Optional[str], Field(description="Project name")] = None
    object_id: ObjectId = None
    description: Annotated[Optional[str], Field(description="Project description")] = None
    key_values: list[KeyValue] = Field(
        default_factory=list,
        description="List of key-value pairs to be added to this project",
        alias="keyValues"
    )
    tag: list[KeyValue] = Field(
        default_factory=list,
        description="List of tags associated with this project"
    )

class ProjectRequest(ProjectFields):
    """
    Fields needed to upload a project to OMERO
    """
    dataset: Annotated[List[DatasetRequest], Field(
        default_factory=list,
        description="List of datasets in this project"
    )]

class UploadFields(TaskQueueBase):
    """
    Common fields for an OMERO upload
    """
    group: Annotated[str, Field(
        description="The OMERO group name"
    )]
    import_user: Annotated[str, Field(
        description="Username of the user importing the data",
        alias="importUser",
        validation_alias=AliasChoices("importUser", "import_user")
    )]

class UploadRequest(UploadFields):
    """
    Data structure used to upload images to OMERO
    """
    project: list[ProjectRequest] = Field(
        description="List of projects to be uploaded",
        default_factory=list,
    )

State = Annotated[Literal[
    "PENDING", "FAILURE", "STARTED", "SUCCESS", "IMPORT-IN-PROGRESS"
], Field(
    description="The status of the upload task"
)]

class UploadReponse(TaskQueueBase):
    """
    Response schema from the /send-task/gs_import_run_omero_import endpoint
    """
    task_id: Annotated[str, Field(
        description="The ID of the task in the task queue",
        alias="task-id",
        validation_alias=AliasChoices("task-id", "task_id")
    )]
    state: State

TimeStamp = Annotated[Optional[datetime], BeforeValidator(
    lambda v: v if v is None else datetime.fromtimestamp(v),
), Field(description="Timestamp of an event")]

class UploadStatus(TaskQueueBase):
    """
    Response schema from the `/api/task/info/{task_id}` endpoint
    """
    uuid: str
    name: str
    state: State
    received: TimeStamp
    sent: None
    started: TimeStamp
    rejected: TimeStamp
    succeeded: TimeStamp
    failed: TimeStamp
    retried: None
    revoked: None
    args: str
    kwargs: str
    eta: None
    expires: None
    retries: int
    worker: str
    result: Optional[str]
    exception: Optional[str]
    timestamp: TimeStamp
    runtime: Optional[float]
    traceback: Optional[str]
    exchange: None
    routing_key: None
    clock: int
    client: None
    root: str
    root_id: str
    parent: None
    parent_id: None
    children: list

class ImportSummary(TaskQueueBase):
    """
    Summary of an image import
    """
    number_of_files: str
    number_of_omero_images: str
    number_of_errors: str
    importer_time: str
    error: None
    exception: None
    file_exists: Optional[bool]
    clientpath: Optional[str]
    plate_id: Annotated[None, Field(
        alias="plateId",
    )]
    image_id: Annotated[OmeroId, Field(
        alias="imageId",
    )]
    time: str

ImportStatus = Annotated[Optional[Literal["SUCCEEDED", "FAILED"]], Field(
    description="The status of part the import task",
    default=None,
    alias="importStatus"
)]

class ImageResponse(ImageRequest):
    """
    Data returned by the task queue after the image has been imported
    """
    import_status: ImportStatus = None
    import_summary: Annotated[Union[ImportSummary, str, None], Field(
        alias="importSummary"
    )] = None
    object_id: ObjectId = None
    fileset_id: Annotated[OmeroId, Field(alias="filesetId")] = None
    error: Error = None

class DatasetResult(DatasetFields):
    """
    Data returned by the task queue after the Dataset has been imported
    """
    image: list[ImageResponse]
    import_status: ImportStatus = None
    error: Error = None

class ProjectResult(ProjectFields):
    """
    Data returned by the task queue after the Project has been imported
    """
    dataset: List[DatasetResult]
    import_status: ImportStatus = None
    error: Error = None

class UploadResult(UploadFields):
    """
    Data returned by the task queue after the Upload object has been imported
    """
    project: List[ProjectResult]
    import_status: ImportStatus = None
    error: Error = None

class UploadResultSet(TaskQueueBase):
    """
    Response schema from the `/api/task/result/{task_id}` endpoint
    """
    task_id: Annotated[str, Field(alias='task-id')]
    state: State
    result: List[UploadResult]
