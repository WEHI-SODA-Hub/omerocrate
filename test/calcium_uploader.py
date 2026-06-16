from omerocrate.uploader import OmeroUploader
from omero import gateway
from rdflib.query import ResultRow
from omero.model import ImageI

class CalciumUploader(OmeroUploader):
    # Capture the acquisition date
    image_query = """
        SELECT ?file_path
        WHERE {
            ?file_path a schema:MediaObject ;
            OPTIONAL{ ?file_path schema:dateModified ?date }
            FILTER (STRAFTER(STR(?file_path), ".") IN ("jpg", "jpeg", "png", "tiff", "tif", "bmp", "gif"))
        }
    """

    def process_image(self, image: gateway.ImageWrapper, result: ResultRow, dataset: gateway.DatasetWrapper) -> None:
        image_obj: ImageI = image._obj
        if 'date' in result:
            # Use the result to set the acquisition date
            image_obj.setAcquisitionDate(result['date'])
        image.save()
