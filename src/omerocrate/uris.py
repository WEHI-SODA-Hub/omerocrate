from rdflib import URIRef

#: If set to true, upload to OMERO
upload = URIRef("https://w3id.org/WEHI-SODA-Hub/omerocrate/upload")

#: Must link to a `MediaObject` that this is the segmentation for
segmentation_for = URIRef("https://w3id.org/WEHI-SODA-Hub/omerocrate/segmentationFor")

#: This links an object to an existing OMERO user group
#: There is an OME RDF scheme, but it doesn't include RDF properties: https://gist.github.com/stefanches7/5b3402331d901bb3c3384bac047c4ac2 for the OME RDF scheme
experimenter_group = URIRef("https://w3id.org/WEHI-SODA-Hub/omerocrate/experimenterGroup")
