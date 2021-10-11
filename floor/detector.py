import io
import json
from typing import List

import numpy as np
import onnxruntime as rt
from PIL import Image

MEAN = [0.6826, 0.6033, 0.5841]
STD = [0.2460, 0.2537, 0.2448]
IMG_SIZE = 640


def json_dump(obj, dst_p):
    with open(dst_p, "w", encoding="utf8") as target:
        json.dump(obj, target)


def pil_loader(src, from_byte=False):
    if from_byte:
        src = io.BytesIO(src)
    im_rgb = Image.open(src).convert("RGB")
    return im_rgb


def preprocess(img: Image.Image, size, torch_nor=True):
    im = np.array(img.resize((size, size))) / 255
    if torch_nor:
        im = (im - MEAN) / STD
    im = np.expand_dims(im.transpose(2, 0, 1), axis=0)
    return im.astype(np.float32)


TARGETS = [(5, "door"), (6, "door"), (7, "window"), (8, "window")]


class Detector:
    IMG_SIZE = 640

    def __init__(
        self,
        yolo_p: str,
    ) -> None:
        self.yolo = rt.InferenceSession(yolo_p)

    def predict(self, im0: Image.Image, prob_thresh=0.2, overlapThresh=0.5):
        pred_onx = self.yolo.run(
            None, {"images": preprocess(im0, self.IMG_SIZE, torch_nor=False)}
        )[0][0]
        _boxes: List[list] = []
        _labels: List[str] = []
        _probs: List[float] = []

        for loc, label in TARGETS:
            filter_idx = np.apply_along_axis(
                lambda row: row[4] * row[loc] > prob_thresh, 1, pred_onx
            )
            pre_boxes = pred_onx[filter_idx]
            if len(pre_boxes) == 0:
                continue
            tmp_boxes_ = np.apply_along_axis(to_boxes, 1, pre_boxes)
            intra_pick = non_max_suppression(tmp_boxes_, overlapThresh)
            tmp_boxes = tmp_boxes_[intra_pick]
            _boxes.extend(tmp_boxes)
            _labels.extend(label for _ in tmp_boxes)
            _probs.extend(float(row[4] * row[loc]) for row in pre_boxes[intra_pick])

        boxes_arr = np.array(_boxes)
        labels = np.array(_labels)
        probs = np.array(_probs)
        inter_pick = non_max_suppression(boxes_arr, overlapThresh)
        boxes_arr = boxes_arr[inter_pick]
        labels = labels[inter_pick]
        probs = probs[inter_pick]

        h0 = im0.height
        w0 = im0.width
        wg = self.IMG_SIZE / w0
        hg = self.IMG_SIZE / h0
        boxes = list(map(lambda x: box_adjust(x, wg, w0, hg, h0), boxes_arr))
        return labels, boxes, probs


def to_boxes(prebox):
    w_half = prebox[2] / 2
    h_half = prebox[3] / 2
    lx = prebox[0] - w_half
    ly = prebox[1] - h_half
    rx = prebox[0] + w_half
    ry = prebox[1] + h_half
    return [lx, ly, rx, ry]


def box_adjust(tmp_box, wg, w0, hg, h0):
    lx = _clip(tmp_box[0] / wg, w0)
    ly = _clip(tmp_box[1] / hg, h0)
    rx = _clip(tmp_box[2] / wg, w0)
    ry = _clip(tmp_box[3] / hg, h0)
    return [int(v) for v in (lx, ly, rx, ry)]


def _clip(v, up, down=0):
    if v < down:
        return down
    if v > up:
        return up
    else:
        return v


# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    return pick
