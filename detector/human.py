""" Created by MrBBS """
# 9/19/2022
# -*-encoding:utf-8-*-

import cv2
import numpy as np
import onnxruntime

onnxruntime.set_default_logger_severity(3)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


class Detector:
    def __init__(self, cuda=False):
        w = "./src/person.onnx"
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(w, providers=providers)
        self.outname = [i.name for i in self.session.get_outputs()]
        self.inname = [i.name for i in self.session.get_inputs()][0]
        self.names = ['head', 'person']

    def detect(self, img):
        image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        inp = {self.inname: im}
        outputs = self.session.run(self.outname, inp)[0]
        heads, bodys = [], []
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            if cls_id == 1:
                bodys.append(box)
            else:
                heads.append(box)
            score = round(float(score), 3)
            name = self.names[cls_id]
            yield box, score, name

    @staticmethod
    def find_head_for_body(ids, bodys, heads):

        people = {}
        for num, box in zip(ids, bodys):
            box = list(map(int, box))
            people.setdefault(num, [box])
            for i, (x1, y1, x2, y2) in enumerate(heads):
                if x2 in range(box[0], box[2]) and y2 in range(box[1], box[3]):
                    people[num].append([x1, y1, x2, y2])
                    heads.pop(i)
        num_nobody = len(heads)
        num_person = len(people.keys())
        if num_nobody > 0:
            for i, box in enumerate(heads):
                people.setdefault(i + num_person + 1, [box])
        return people
