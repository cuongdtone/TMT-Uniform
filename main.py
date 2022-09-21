# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/20/2022

import time
import cv2
from bytetrack.bytetracker import ByteTrack
from detector import human
from utils.crowd_tracking import Crowd
from utils.draw import full_draw

st = time.time()
# cv2.namedWindow('cc', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
tracker = ByteTrack(30, track_thresh=0)
print('Load source ... ', time.time() - st)

st = time.time()
human_detector = human.Detector(cuda=False)
crowd_tracker = Crowd()
print('Load model ... ', time.time() - st)
fpss = []

while cap.isOpened():
    ret, frame = cap.read()
    ori = frame.copy()
    st = time.time()
    full_body_bboxes = []
    heads = []
    class_ids = []
    scores = []

    for box, score, name in human_detector.detect(frame):
        if name == 'person':
            full_body_bboxes.append([*box, 0.9])
            class_ids.append(1)
            scores.append(0.9)
        else:
            heads.append(box)

    # full_body_bboxes = np.array(full_body_bboxes)
    ids, full_body_bboxes, _, _ = tracker(frame, full_body_bboxes, scores, class_ids)

    people = human_detector.find_head_for_body(ids, full_body_bboxes, heads)
    # people is dict: key is tracking id, value is [body_box, head_box]
    people = crowd_tracker.update(ori, people)
    # print(people)
    frame = full_draw(frame, people)

    fps = 1/(time.time() - st)
    fpss.append(fps)
    cv2.putText(frame, str(int(sum(fpss)/len(fpss))), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('cc', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

