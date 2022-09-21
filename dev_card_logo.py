# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/20/2022
import time

import cv2
from detector import card_logo

card_logo_detector = card_logo.Detector(cuda=False)

st = time.time()
cap = cv2.VideoCapture(0)  # r'C:\Users\Cuong Tran\Pictures\Camera Roll\WIN_20220919_16_50_31_Pro.mp4')
print('loading source ..', time.time() - st)
while cap.isOpened():
    ret, frame = cap.read()
    ori = frame.copy()
    for box, name, score in card_logo_detector.detect(frame):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('cc', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
