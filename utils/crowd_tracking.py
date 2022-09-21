# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/20/2022
import time

import yaml
import cv2
import numpy as np
from detector import human, card_logo
from utils.face_landmark import TDDFA_ONNX
from utils.face_recognizer import ArcFaceONNX
from utils.sqlite3_db import load_data, insert_check
import base64

landmarks_detector = TDDFA_ONNX(**(yaml.load(open('src/mb1_120x120.yml'), Loader=yaml.SafeLoader)))
recognizer = ArcFaceONNX('src/w600k_mbf.onnx')
card_logo_detector = card_logo.Detector(cuda=False)
face_data = load_data()


class Person(object):
    def __init__(self, track_id, image):
        self.id = track_id
        self.name = None
        self.id_code = None
        self.have_logo = 0
        self.have_card = 0
        self.time_to_live = 0  # frame counting
        self.last_time = time.time()
        self.image = image

    def analysis(self, image, head, full_body):
        """
        image is frame
        head box
        body
        """
        self.last_time = time.time()
        self.time_to_live += 1
        if self.time_to_live < 100:
            try:
                x1_tb, y1_tb, x2_tb, y2_tb = full_body
                person_img = image[y1_tb:y2_tb, x1_tb:x2_tb, :]
                person_img2 = person_img.copy()
                cv2.imencode('.jpg', person_img2)
                self.image = person_img2
            except:
                pass

        body = self.crop_body(full_body, head)
        x1_tb, y1_tb, x2_tb, y2_tb = body
        # phase 1: logo and card
        person_img = image[y1_tb:y2_tb, x1_tb:x2_tb, :]
        boxes = []
        clss_ids = []
        try:
            for box, name, score in card_logo_detector.detect(person_img):
                x1, y1, x2, y2 = box
                offset_box  = (x1 + x1_tb, y1 + y1_tb, x1_tb + x2, y2 + y1_tb)
                boxes.append(offset_box)
                clss_ids.append(name)
                if name == 'logo':
                    self.have_logo += 1
                elif name == 'card':
                    self.have_card += 1
        except Exception as e:
            # print(e)
            pass
        # phase 2: Identity
        if self.name is None:
            landmark = landmarks_detector(image, [head])[0]
            landmark = landmark.T[:, [0, 1]].astype('int')
            kps = np.array([landmark[30], landmark[45], landmark[30], landmark[48], landmark[54]])
            feat = recognizer.face_encoding(image, kps)
            info = recognizer.face_compare(feat, face_data)
            self.name = info['fullname']
            self.id_code = info['code']

        return (x1_tb, y1_tb, x2_tb, y2_tb), (boxes, clss_ids), self.name  # body, accessory and name

    @staticmethod
    def crop_body(full_body, head):
        x1_h, y1_h, x2_h, y2_h = head
        # x1_b, y1_b, x2_b, y2_b = full_body
        y_chin = y2_h
        center_head = (x1_h + (x2_h - x1_h) // 2, y1_h + (y2_h - y1_h) // 2)
        h_top_body = int((y2_h - y1_h) * 2)
        w_shoulder = x2_h - x1_h
        x1_tb, y1_tb, x2_tb, y2_tb = (
            center_head[0] - w_shoulder, y_chin, center_head[0] + w_shoulder, y_chin + h_top_body)
        return x1_tb, y1_tb, x2_tb, y2_tb
        # cv2.rectangle(frame, (x1_tb, y1_tb), (x2_tb, y2_tb), (0, 0, 255), 2)


class Crowd(object):
    def __init__(self):
        self.people = {}  # contain {id: person}
        self.ids = []
        self.delete_person_delay = 1  # tracking will cancel in 3s

    def update(self, frame, people: dict):
        # people is dict: key is tracking id, value is [body_box, head_box]
        result_track = {}
        for track_id in people.keys():
            if len(people[track_id]) == 2:
                full_body, head = people[track_id]
                full_body = list(map(int, full_body))
                head = list(map(int, head))

                if track_id not in self.people.keys():
                    self.people.update({track_id: Person(track_id, frame)})
                    body, accessory, name = self.people[track_id].analysis(frame, head, full_body)
                else:
                    body, accessory, name = self.people[track_id].analysis(frame, head, full_body)
                #  accessory is (boxes, class name) of card and logo
                result_track.update({track_id: {'full_body': full_body,
                                                'body': body,
                                                'head': head,
                                                'accessory': accessory,
                                                'name': name}})
        self.delete_tracking()
        return result_track

    def delete_tracking(self):
        now = time.time()
        keys = self.people.keys()
        for i in keys:
            person: Person = self.people[i]
            if now - person.last_time > self.delete_person_delay:
                # write to db
                card_count = person.have_card
                logo_count = person.have_logo
                name = person.name
                code = person.id_code
                ttl = person.time_to_live
                image = person.image

                self.voting_and_write_db(image, card_count, logo_count, ttl, name, code)
                # print(f'Person {name} live {ttl}s, Counted {card_count} card, {logo_count} logo')
                # write ending
                self.people = self.remove_key(self.people, i)

    @staticmethod
    def voting_and_write_db(frame, card, logo, ttl, name, code):
        """
        frame bg opencv
        """
        ratio = 2  # ratio of vote decided have card or logo
        card = True if card > ttl/ratio else False
        logo = True if logo > ttl / ratio else False
        retval, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        insert_check(None, code, name, card, logo, jpg_as_text)

    @staticmethod
    def remove_key(d, key):
        r = dict(d)
        del r[key]
        return r


