import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def parse_XML(xml_p: str, img_dir: Path, label_dir: Path):
    src_p = Path(xml_p)

    img_p = src_p.with_suffix(".tiff")
    if not img_p.exists():
        return
    img_ori = Image.open(img_p).convert("P").convert("RGB")
    label_file = open(
        label_dir / f"{src_p.parent.name}{src_p.stem}.txt", "w", encoding="utf8"
    )
    w0, h0 = img_ori.size
    img = resize_byH(img_ori)
    img.save(img_dir / f"{src_p.parent.name}{src_p.stem}.jpg")

    tree = ET.parse(xml_p)
    root = tree.getroot()
    labels = root.findall("./ov/o/gom.std.OSymbol")
    if labels is None:
        return
    for child in labels:
        attrs = child.attrib
        name = attrs["label"]
        if "door" in name or "window" in name:
            x0 = float(attrs["x0"])
            y0 = float(attrs["y0"])
            x1 = float(attrs["x1"])
            y1 = float(attrs["y1"])
            width = x1 - x0
            height = y1 - y0
            x_center = x0 + (width / 2)
            y_center = y0 + (height / 2)
            yolo_attrs = (
                _label_num(name),
                x_center / w0,
                y_center / h0,
                width / w0,
                height / h0,
            )
            label_file.write(" ".join(str(attr) for attr in yolo_attrs))
            label_file.write("\n")

    label_file.close()


def _label_num(label):
    if "door1" in label:
        return "0"
    elif "door2" in label:
        return "1"
    elif "window1" in label:
        return "2"
    elif "window2" in label:
        return "3"
    else:
        print(label)


def resize_byH(img: Image.Image, new_height=1280):
    width, height = img.size
    ratio = width / height
    new_width = int(ratio * new_height)
    return img.resize((new_width, new_height), Image.ANTIALIAS)


def parse_XMLs(src, dst: Path):
    batchs = Path(src).iterdir()
    for idx, batch in enumerate(batchs, start=1):
        if idx >= 9:
            _dst = dst / "val"
            _im_dir = _dst / "images"
            _label_dir = _dst / "labels"
            _im_dir.mkdir(exist_ok=True, parents=True)
            _label_dir.mkdir(exist_ok=True, parents=True)
        else:
            _dst = dst / "train"
            _im_dir = _dst / "images"
            _label_dir = _dst / "labels"
            _im_dir.mkdir(exist_ok=True, parents=True)
            _label_dir.mkdir(exist_ok=True, parents=True)
        for xml in batch.glob("*.xml"):
            parse_XML(str(xml), _im_dir, _label_dir)
