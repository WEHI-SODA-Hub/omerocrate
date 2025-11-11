from rdflib import URIRef

upload = URIRef("https://w3id.org/WEHI-SODA-Hub/omerocrate/upload")
"""
Turns on or off OMERO upload
The subject of this predicate should be an entity representing an image file
The object should be a boolean literal
"""

segmentation_for = URIRef("https://w3id.org/WEHI-SODA-Hub/omerocrate/segmentationFor")
"""
Connects a segmentation to the image it segments
The subject of this predicate should be an entity representing a segmentation
The object should be the image (typically a `MediaObject`) that this segmentation is for
"""

experimenter_group = URIRef(
    "https://w3id.org/WEHI-SODA-Hub/omerocrate/experimenterGroup"
)
"""
Customizes the OMERO experimenter group to own the dataset
The subject of this predicate should be the root `Dataset` of the crate
The object should be a string literal of the OMERO group name to upload into
"""
