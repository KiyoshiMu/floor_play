from pathlib import Path
from zipfile import ZipFile


def unzip_target(zip_p, dst: Path):

    with ZipFile(zip_p, "r") as docs:
        for fn in docs.namelist():
            if "tiff" in fn or "xml" in fn:
                docs.extract(fn, dst)


def unzip_targets(src, dst: Path):
    for zip_p in Path(src).glob(".zip"):
        unzip_target(zip_p, dst)
