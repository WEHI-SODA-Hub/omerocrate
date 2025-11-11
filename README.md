# omeROcrate

Integration layer between the OMERO image platform and RO-Crate metadata standard.

## Installation

These instructions assume you're using [uv](https://docs.astral.sh/uv/).

Create a project if you haven't already:

```bash
uv init
```

You may optionally want to install [Glencoe's prebuilt binaries](https://www.glencoesoftware.com/blog/2023/12/08/ice-binaries-for-omero.html), e.g.
```bash
uv pip install https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp39-cp39-manylinux_2_28_x86_64.whl
```

Then install using:
```bash
uv add git+https://github.com/WEHI-SODA-Hub/OmeroCrate
```

## Authentication

You will need to set the following environment variables:

```bash
export OMERO_PASSWORD=xxx
export OMERO_HOST=xxx
```

There are two other optional variables:

- `OMERO_USER` defaults to the current username
- `OMERO_PORT` defaults to 4064

## Simple Example

With all the setup done, we can upload some data to OMERO.

In this example we will use the Calcium Imaging data thanks to:

> Schröder, M., Staehlke, S., Groth, P., Nebe, J. B., Spors, S., & Krüger, F. (2022). Structure-based knowledge acquisition from electronic lab notebooks for research data provenance documentation. Journal of Biomedical Semantics, 13. https://doi.org/10.1186/s13326-021-00257-x

But you can use your own data instead.

First we obtain the data:

```bash
git clone https://github.com/SFB-ELAINE/Ca-imaging-RO-Crate.git
```

Then we do a simple upload:

```bash
uv run omerocrate upload Ca-imaging-RO-Crate/
```

omeROcrate can do basic crate uploads without configuration.

## Advanced Example

RO-Crates are very flexible, which means that you probably use some extra types and properties that aren't handled by omeROcrate by default.
Not to worry, you can quite easily customize it to suit your crates, via subclassing.

For example, in the calcium imaging data, there are some entities that look a bit like this:
```json
{
    "@id": "Data/02_Bild-nach-Stimulation_5V_7.9Hz.jpg",
    "@type": "File",
    "https://schema.org/dateModified": {
        "@type": "xsd:dateTime",
        "@value": "2021-03-03T16:58:35"
    }
}
```

When customizing omeROcrate, it is best to first look at [OMERO's own metadata schema](https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome.html) and decide if your metadata fits into one of the existing parts of the schema.
If not, the metadata can always become an annotation.

In this case, `dateModified` could be interpreted as the acquisition date of the image.
If this is true, we could subclass `OmeroUploader` to handle this properly.
[You can find an example of this subclassing here](https://github.com/WEHI-SODA-Hub/OmeroCrate/blob/main/test/calcium_uploader.py).

Currently if you subclass `OmeroUploader`, you will have to perform the upload in Python:

```python
uploader = CalciumUploader(
    conn=connection,
    crate=ca_imaging_1021
)
dataset = uploader.execute()
```
