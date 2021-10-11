from pathlib import Path

from PIL import ImageDraw, ImageFont

from floor.detector import Detector, json_dump, pil_loader


def main(img_p="data/file_0.jpg"):
    font_family = "arial.ttf"
    detector = Detector("yolos.v0.onnx")

    im0 = pil_loader(img_p, from_byte=False)
    im_draw = ImageDraw.Draw(im0)
    font = ImageFont.truetype(font_family, 22)
    ret = []
    for label, box, prob in zip(*detector.predict(im0)):
        im_draw.rectangle(box, width=5, outline="red")
        im_draw.text(
            box[:2], f"{label}:{prob:.02f}", stroke_width=1, fill="green", font=font
        )
        ret.append(dict(label=label, box=box, prob=prob))
    fn = Path(img_p).stem
    im0.save(f"{fn}_ret.jpg")
    json_dump(ret, f"{fn}_ret.json")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
