import asyncio
from pathlib import Path
import ssl
from typing import AsyncIterable, Any, Iterable
from omero import gateway, model
import httpx
import os

from pydantic import Field, computed_field, ValidationError
import truststore

from omerocrate.taskqueue import models as upload_models
from omerocrate.uploader import OmeroUploader
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


def log_request(request: httpx.Request) -> None:
    """
    Logs the request details.
    """
    logger.debug(f"Request: {request.method} {request.url}")
    logger.debug(f"Request headers: {request.headers}")
    logger.debug(
        f"Request body: {request.content.decode('utf-8') if request.content else None}"
    )


def log_response(response: httpx.Response, request: bool = True) -> None:
    """
    Logs the response details.
    """
    logger.debug(f"Response: {response.status_code} {response.url}")
    logger.debug(f"Response headers: {response.headers}")
    logger.debug(f"Response body: {response.text}")
    if request:
        log_request(response.request)


class TaskqueueUploader(OmeroUploader):
    """
    Subclass of OmeroUploader that uses `gs-taskqueue` to upload images to OMERO.
    """

    host: str = Field(
        default_factory=lambda: os.environ["FLOWER_HOST"],
        description="Host of the Flower API server",
    )
    username: str = Field(
        default_factory=lambda: os.environ["FLOWER_USER"],
        description="Username for the Flower API server",
    )
    password: str = Field(
        default_factory=lambda: os.environ["FLOWER_PASSWORD"],
        description="Password for the Flower API server",
    )

    @computed_field
    @property
    def client(self) -> httpx.AsyncClient:
        """
        Returns an authenticated HTTP client for the Flower API server.
        """
        return httpx.AsyncClient(
            auth=(self.username, self.password),
            # Support local certificate stores
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        )

    async def upload_images(
        self, image_paths: list[Path], dataset: gateway.DatasetWrapper, **kwargs: Any
    ) -> AsyncIterable[gateway.ImageWrapper]:
        # We create a dummy project, because `gs-taskqueue` requires one.
        # This makes it easy delete it afterwards
        project = gateway.ProjectWrapper(self.conn, model.ProjectI())
        project.setName(str(uuid4()))
        project.save()
        project._linkObject(dataset, "ProjectDatasetLinkI")

        req = upload_models.UploadRequest(
            project=[
                upload_models.ProjectRequest(
                    object_id=project.getId(),
                    dataset=[
                        upload_models.DatasetRequest(
                            object_id=dataset.getId(),
                            image=[
                                upload_models.ImageRequest(
                                    name=str(uuid4()), file_path=str(path)
                                )
                                for path in image_paths
                            ],
                        )
                    ],
                )
            ],
            # Upload into the same group as the dataset
            group=dataset.getDetails().getGroup().getName(),
            import_user=os.environ["OMERO_USER"],
        )

        task_id = (await self.upload_to_omero(req)).task_id
        # Loop until the task is finished
        while True:
            await asyncio.sleep(1)
            status = await self.upload_status(task_id)
            if status.state in {"SUCCESS", "FAILURE"}:
                break
        result = await self.upload_result(task_id)
        errors = list(self.error_from_result(result))
        if len(errors) > 0:
            error_messages = "\n".join(errors)
            raise Exception(
                f"Upload failed: {error_messages}\nFull JSON\n: {result.model_dump_json(indent=4)}"
            )

        # Unlink and delete the project, since we never wanted it
        for link in project.getChildLinks():
            if isinstance(link._obj, model.ProjectDatasetLinkI):
                self.conn.deleteObject(link._obj)
        self.conn.deleteObject(project._obj)

        # Get in-memory wrappers for each resulting image
        for upload in result.result:
            for project in upload.project:
                for result_dataset in project.dataset:
                    for image in result_dataset.image:
                        if isinstance(image.object_id, list):
                            for object_id in image.object_id:
                                yield self.conn.getObject("Image", object_id)
                        elif image.object_id is not None:
                            yield self.conn.getObject("Image", image.object_id)

    def error_from_result(self, result: upload_models.UploadResultSet) -> Iterable[str]:
        """
        Returns an error message if the upload failed, or None if it succeeded.
        """
        if result.state != "SUCCESS":
            yield "Entire upload failed."
        for upload in result.result:
            if upload.import_status == "FAILED" or upload.error is not None:
                yield f"Singular upload failed with error {upload.error}"
            for project in upload.project:
                for dataset in project.dataset:
                    if dataset.import_status == "FAILED" or dataset.error is not None:
                        yield f"Dataset upload failed with error {dataset.error}."
                    for image in dataset.image:
                        if image.import_status == "FAILED" or image.error is not None:
                            yield f"Image {image.file_path} upload failed with error {image.error}."

    async def upload_to_omero(
        self, upload: upload_models.UploadRequest
    ) -> upload_models.UploadReponse:
        """
        Submits the upload request to the Flower API server and returns the API response.
        """
        response = await self.client.post(
            f"{self.host}/flower/api/task/send-task/gs_import_run_omero_import",
            json={"args": [upload.model_dump(exclude_none=True)]},
        )
        log_response(response, request=True)
        response.raise_for_status()
        return upload_models.UploadReponse.model_validate(response.json())

    async def upload_status(self, task_id: str) -> upload_models.UploadStatus:
        """
        Checks the status of an upload task.
        """
        URL_CHECK_INFO = f"{self.host}/flower/api/task/info/{task_id}"

        # Check the task status
        response = await self.client.get(
            URL_CHECK_INFO,
        )
        log_response(response, request=True)
        response.raise_for_status()
        return upload_models.UploadStatus.model_validate(response.json())

    async def upload_result(self, task_id: str) -> upload_models.UploadResultSet:
        """
        Gets the result of a completed upload task.
        """
        response = await self.client.get(
            f"{self.host}/flower/api/task/result/{task_id}",
        )
        log_response(response, request=True)
        response.raise_for_status()

        try:
            return upload_models.UploadResultSet.model_validate(response.json())
        except ValidationError as e:
            raise Exception(f"Upload failed. Full JSON:\n{response.text}") from e
