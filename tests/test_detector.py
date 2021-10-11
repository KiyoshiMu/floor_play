from PIL import ImageDraw, ImageFont

from floor.detector import Detector, pil_loader


def test_detector():
    inputs = "data/file_0.jpg"
    font_family = "../arial.ttf"
    detector = Detector("data/yolos.v0.onnx")
    im0 = pil_loader(inputs, from_byte=False)
    im_draw = ImageDraw.Draw(im0)
    font = ImageFont.truetype(font_family, 22)
    for label, box, prob in zip(*detector.predict(im0)):
        im_draw.rectangle(box, width=5, outline="red")
        im_draw.text(
            box[:2], f"{label}:{prob:.02f}", stroke_width=1, fill="green", font=font
        )
    im0.save("exp.jpg")
