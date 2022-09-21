# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/19/2022

from openpyxl import load_workbook
import yaml
import cv2
import numpy as np
from detector import human
from utils.face_landmark import TDDFA_ONNX
from utils.face_recognizer import ArcFaceONNX
from utils.sqlite3_db import insert_employee, connect_database


landmarks_detector = TDDFA_ONNX(**(yaml.load(open('src/mb1_120x120.yml'), Loader=yaml.SafeLoader)))
recognizer = ArcFaceONNX('src/w600k_mbf.onnx')

human_detector = human.Detector(cuda=False)

id_code = input('ID: ')
name = input('Name: ')

frame = cv2.imread(r'C:\Users\Cuong Tran\Pictures\Camera Roll\WIN_20220919_17_06_44_Pro.jpg')
ori = frame.copy()

for box, score, clss in human_detector.detect(frame):

    if clss == 'head':
        head = box
        cv2.rectangle(frame, head[:2], head[2:], (0, 255, 0), 2)
        landmark = landmarks_detector(frame, [head])[0]
        landmark = landmark.T[:, [0, 1]].astype('int')
        kps = np.array([landmark[30], landmark[45], landmark[30], landmark[48], landmark[54]])
        feat = recognizer.face_encoding(ori, kps)
        cv2.polylines(frame, [kps], True, color=(0, 255, 255), thickness=2)


db = connect_database()
insert_employee(db, id_code, name, str(feat))
