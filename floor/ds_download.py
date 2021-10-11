from pathlib import Path

import requests

LINK = "http://mathieu.delalandre.free.fr/projects/sesyd/symbols/floorplans/floorplans16-{:02d}.zip"


def download(url: str, dst: Path):
    fp = dst / url.rsplit("/", 1)[-1]
    if fp.exists():
        return
    response = requests.get(url, stream=True)
    with open(fp, "wb") as handle:
        for data in response.iter_content(chunk_size=1024):
            handle.write(data)


def download_ds():
    dst = Path("zips")
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(1, 11):
        url = LINK.format(i)
        download(url, dst)


if __name__ == "__main__":
    for i in range(1, 11):
        url = LINK.format(i)
        download(url, Path("data"))
        break
