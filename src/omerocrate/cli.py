from __future__ import annotations
from pathlib import Path
from typing import Annotated
import typer
from importlib import import_module
from asyncio import run
from omerocrate.gateway import from_env
from omerocrate.uploader import OmeroUploader, SegmentationUploader

app = typer.Typer(help="CLI for uploading RO-Crates to OMERO")

@app.command(help="Upload an RO-Crate to OMERO")
def upload(
    crate: Annotated[Path, typer.Argument(help="Path to the directory containing the RO-Crate")],
    uploader_path: Annotated[str, typer.Option("--uploader", "-u", help="Module path to the OmeroUploader class")] = "omerocrate.ApiUploader",
    seg_uploader_path: Annotated[str, typer.Option("--seg-uploader", "-s", help="Module path to the SegmentationUploader class")] = "omerocrate.SegmentationUploader",
    seg_upload_dir: Annotated[Path | None, typer.Option("--seg-upload-dir", "-d", help="Directory to upload segmentations")] = None
):
    module_path, cls_name = uploader_path.rsplit('.', 1)
    module = import_module(module_path)
    uploader_cls = getattr(module, cls_name)
    if not issubclass(uploader_cls, OmeroUploader):
        raise typer.BadParameter(f"{uploader_path} is not a valid OmeroUploader class")

    seg_module_path, seg_cls_name = seg_uploader_path.rsplit('.', 1)
    seg_module = import_module(seg_module_path)
    seg_uploader_cls = getattr(seg_module, seg_cls_name)
    if not issubclass(seg_uploader_cls, SegmentationUploader):
        raise typer.BadParameter(f"{seg_uploader_path} is not a valid SegmentationUploader class")

    conn = from_env()
    seg_uploader = seg_uploader_cls(conn=conn, upload_directory=seg_upload_dir)
    uploader = uploader_cls(
        conn=conn,
        crate=crate,
        segmentation_uploader=seg_uploader
    )
    dataset = run(uploader.execute())
    # stderr
    typer.echo(f"Uploaded dataset with ID: {dataset.id} and name: {dataset.name}", err=True)
    # stdout
    typer.echo(dataset.id)

def main():
    app()
