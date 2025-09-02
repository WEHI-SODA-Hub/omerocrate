from dataclasses import dataclass
import logging
from pathlib import Path
from time import sleep
from typing import Any, Iterable, Literal, cast, AsyncIterable
from rdflib import Graph, URIRef
from rdflib.query import ResultRow
from rdflib.term import Identifier
from functools import cached_property
from omero import model, gateway, grid, cmd
from omero.model import enums
from omero.rtypes import rstring, rbool
from urllib.parse import urlparse
import asyncio
from pydantic import BaseModel

from omerocrate.utils import user_in_group

logger = logging.getLogger(__name__)

Namespaces = dict[str, URIRef]
Variables = dict[str, Identifier]

class OmeroUploader(BaseModel, arbitrary_types_allowed=True):
    """
    Class that handles the conversion between RO-Crate metadata and OMERO objects.
    Users are encouraged to subclass this and override any of the public methods to customize the behavior.
    Refer to the method documentation for more information.
    """
    conn: gateway.BlitzGateway
    "OMERO connection object, typically obtained using [`from_env`][omerocrate.gateway.from_env]"
    crate: Path
    "Path to the directory containing the crate"
    transfer_type: Literal["ln", "ln_s", "ln_rn", "cp", "cp_rm", "upload", "upload_rm"] = "upload"
    """
    Transfer method, which determines how images are sent to OMERO.
    `ln_s` is "in-place" importing, but it requires that this process has acess to both the image and permissions to write to the OMERO server.
    """

    @property
    def namespaces(self) -> Namespaces:
        """
        Namespaces/prefixes used in all SPARQL queries.
        Override this to add or adjust prefixes, e.g. if you are using additional vocabularies.
        """
        return {
            "schema": URIRef("http://schema.org/"),
            "crate": URIRef(f"{self.crate.as_uri()}/"),
            "omerocrate": URIRef("https://w3id.org/WEHI-SODA-Hub/omerocrate/"),
            "ome": URIRef("http://www.openmicroscopy.org/Schemas/OME/2016-06/"),
        }

    @cached_property
    def graph(self) -> Graph:
        """
        RO-Crate metadata as an RDF graph.
        Typically you don't need to override this method.
        """
        return Graph().parse(source=self.crate / "ro-crate-metadata.json", format='json-ld')

    def select_many(self, query: str, namespaces: Namespaces = {}, variables: Variables = {}) -> Iterable[ResultRow]:
        """
        Helper method for running a SPARQL query on the RO-Crate metadata that returns multiple results.
        Typically you don't need to override this method.
        """
        result = self.graph.query(
            query,
            initNs={
                **self.namespaces,
                **namespaces
            },
            initBindings={
                **variables
            }
        )
        if not result.type == "SELECT":
            raise ValueError("Only SELECT queries are supported")
        return cast(Iterable[ResultRow], result)

    def select_one(self, query: str, namespaces: Namespaces = {}, variables: Variables = {}) -> ResultRow:
        """
        Helper method for running a SPARQL query on the RO-Crate metadata that should return exactly one result.
        Typically you don't need to override this method.
        """
        result = list(self.select_many(query, namespaces, variables))
        if len(result) != 1:
            raise ValueError(f"Expected exactly one result, but got {len(result)}")
        return result[0]

    @cached_property
    def root_dataset_id(self) -> Identifier:
        """
        Returns the ID of the root dataset in the crate.
        You shouldn't need to override this method as this function should work for any conformant RO-Crate.
        """
        result = self.select_one("""
            SELECT ?dataset_id
            WHERE {
                ?dataset_id a schema:Dataset .
                crate:ro-crate-metadata.json schema:about ?dataset_id .
            }
        """)
        return result['dataset_id']

    def find_images(self) -> Iterable[tuple[Identifier, Path]]:
        """
        Finds images that should be uploaded to OMERO.
        Can be overridden to customize the image selection, although this typically isn't needed.
        """
        for result in self.select_many("""
            SELECT ?file_path
            WHERE {
                ?file_path a schema:MediaObject ;
                    omerocrate:upload true ;
            }
        """):
            file_path = result['file_path']
            yield file_path, Path(urlparse(file_path).path)

    def make_dataset(self, group: gateway.ExperimenterGroupWrapper) -> gateway.DatasetWrapper:
        """
        Creates the OMERO dataset wrapper that corresponds to this crate.
        Override to customize the dataset creation.
        """
        dataset = gateway.DatasetWrapper(self.conn, model.DatasetI())

        result = self.select_one("""
            SELECT ?name ?description
            WHERE {
                ?root schema:name ?name .
                ?root schema:name ?description .
            }
        """, variables={"root": self.root_dataset_id})

        # Set the group name for the session, so that the dataset is created in the correct group
        dataset.setName(result['name'])
        dataset.setDescription(result['description'])
        dataset.save()
        return dataset

    async def upload_images(self, image_paths: list[Path], dataset: gateway.DatasetWrapper, **kwargs: Any) -> AsyncIterable[gateway.ImageWrapper]:
        """
        Queries the metadata crate for images and uploads them to OMERO.
        Ideally minimal or no metadata should be set here.
        Images that get yielded should already be saved to the database.

        Params:
            image_paths: List of paths to the images to be uploaded.
            dataset: The OMERO dataset to which the images should be added.
        """
        # Note: Metadata is not set here because we want to allow `process_image()` to be independent of the upload method.

        # Hack to make this method an async generator
        if False:
            yield
        raise NotImplementedError("upload_images() must be implemented in a subclass")

    def connect(self):
        """
        Connects to the OMERO server.
        """
        if not self.conn.isConnected():
            result = self.conn.connect()
            if not result:
                raise ValueError(f"Could not connect to OMERO: {self.conn.getLastError()}")

    def add_image_to_dataset(self, dataset: gateway.DatasetWrapper, image: gateway.ImageWrapper) -> None:
        dataset._linkObject(image, "DatasetImageLinkI")

    def path_from_image_result(self, result: ResultRow) -> Path:
        """
        Converts a SPARQL result row to a Path object.
        """
        return Path(urlparse(result['file_path']).path)

    def process_image(self, uri: URIRef, image: gateway.ImageWrapper) -> None:
        """
        Adds metadata to the image object from the crate.
        Can be overridden to add custom metadata.
        """
        result = self.select_one("""
            SELECT *
            WHERE {
                OPTIONAL {
                    ?file_path schema:name ?name .
                }
                OPTIONAL {
                    ?file_path schema:description ?description .
                }
            }
        """, variables={"file_path": uri})
        if (description := result.description) is not None:
            image.setDescription(str(description))
        if (name := result.name) is not None:
            image.setName(str(name))

        image.save()

    def get_group_name(self) -> str:
        """
        Get the name of the experimenter group from the crate metadata.
        If missing, the dataset name will be used as the group name.

        This probably doesn't need to be overridden.
        """
        try:
            result = self.select_one("""
                SELECT ?group_name
                WHERE {
                    ?root omerocrate:experimenterGroup ?group_name .
                }
            """, variables={"root": self.root_dataset_id})
            return str(result['group_name'])
        except ValueError:
            # If the group name is not specified, use the dataset name
            result = self.select_one("""
                SELECT ?name
                WHERE {
                    ?root schema:name ?name .
                }
            """, variables={"root": self.root_dataset_id})
            return str(result['name'])

    async def make_group(self) -> gateway.ExperimenterGroupWrapper:
        """
        Creates the OMERO experimenter group that corresponds to this crate.
        """
        group_name = self.get_group_name()
        admin_service = self.conn.getAdminService()

        for existing_group in self.conn.listGroups():
            # If the group already exists, add the user to it
            if group_name == existing_group.getName():
                if not user_in_group(self.conn.getUser(), existing_group, admin_service):
                    admin_service.addGroups(self.conn.getUser()._obj, [model.ExperimenterGroupI(existing_group.getId(), False)])

                logger.warning(f"Group {group_name} already exists, using it")
                return existing_group
        else:
            group_id = self.conn.createGroup(
                name=group_name,
                member_Ids=[self.conn.getUser().getId()],
                ldap=False
            )
            return self.conn.getObject("ExperimenterGroup", group_id)
            
        return group
        
    async def execute(self) -> gateway.DatasetWrapper:
        """
        Runs the entire processing workflow.
        Typically you don't need to override this method.
        """
        self.connect()
        img_uris: list[URIRef]
        img_paths: list[Path]
        group = await self.make_group()
        # It seems like the best way to ensure all objects are created in the correct group
        # is to set the group for the session
        self.conn.setGroupForSession(group.getId())
        dataset = self.make_dataset(group)
        img_uris, img_paths = list(zip(*self.find_images()))
        img_wrappers = [img async for img in self.upload_images(img_paths, dataset)]
        for wrapper, uri in zip(img_wrappers, img_uris):
            self.process_image(uri, wrapper)
        return dataset

class ApiUploader(OmeroUploader):
    """
    Subclass of OmeroUploader that uses the OMERO API to upload images.
    """
    async def upload_images(self, image_paths: list[Path], dataset: gateway.DatasetWrapper, *, chunk_size: int = 4096, **kwargs: Any) -> AsyncIterable[gateway.ImageWrapper]:
        handles: list[cmd.HandlePrx] = []
        client = self.conn.c
        repo = client.getManagedRepository()
        algorithm = model.ChecksumAlgorithmI()
        algorithm.setValue(rstring(enums.ChecksumAlgorithmSHA1160))
        for path in image_paths:
            # Fileset, entry and upload entities are required for uploading
            fileset = model.FilesetI()
            entry = model.FilesetEntryI()
            entry.setClientPath(rstring(path))
            fileset.addFilesetEntry(entry)
            upload = model.UploadJobI()
            fileset.linkJob(upload)

            importer = repo.importFileset(
                fileset,
                grid.ImportSettings(
                    checksumAlgorithm=algorithm,
                    doThumbnails=rbool(True),
                    noStatsInfo=rbool(False)
                )
            )
            upload_file = importer.getUploader(0)
            offset = 0
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    upload_file.write(chunk, offset, len(chunk))
                    offset += len(chunk)
            upload_file.close()
            handles.append(importer.verifyUpload([client.sha1(path)]))

        # Wait for the upload to finish
        while handles:
            await asyncio.sleep(0.1)
            for handle in handles:
                response = handle.getResponse()
                if response is not None:
                    handles.remove(handle)
                    pixels: model.PixelsI
                    for pixels in response.pixels:
                        wrapper = gateway.ImageWrapper(conn=self.conn, obj=pixels.getImage())
                        # Add the image to the dataset
                        dataset._linkObject(wrapper, "DatasetImageLinkI")
                        yield wrapper


class SegmentationUploader(ApiUploader):
    """
    Subclass of OmeroUploader that also uploads segmentation masks.
    This class is just a temporary prototype to get this feature working.
    """

    def find_images_with_segmentation(self) -> Iterable[tuple[Identifier, Path, Path]]:
        """
        Finds images containing segmentation masks that should be uploaded to OMERO.
        Can be overridden to customize the query.
        """
        for result in self.select_many("""
            SELECT ?file_path ?segmentation_file
            WHERE {
                ?file_path a schema:MediaObject ;
                    omerocrate:upload true ;
                    omerocrate:segmentationFor ?segmentation_file .
            }
        """):
            file_path = result['file_path']
            segmentation_file = result['segmentation_file']
            yield file_path, Path(urlparse(file_path).path), Path(urlparse(segmentation_file).path)

    async def execute(self) -> gateway.DatasetWrapper:
        """
        Runs the entire processing workflow.
        We're overwriting this method to skip group creation for now.
        """
        self.connect()
        img_uris: list[URIRef]
        img_paths: list[Path]
        seg_paths: list[Path]

        # Skip any group creation for now as test user needs the correct permissions
        group = self.conn.getGroupFromContext()
        dataset = self.make_dataset(group)

        img_uris, img_paths, seg_paths = list(zip(*self.find_images_with_segmentation()))
        img_wrappers = [img async for img in self.upload_images(img_paths, dataset)]
        for wrapper, uri, seg in zip(img_wrappers, img_uris, seg_paths):
            self.process_image(uri, wrapper)
            # Handle segmentation upload here

        return dataset
