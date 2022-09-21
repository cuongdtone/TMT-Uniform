# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/20/2022

import cv2


def full_draw(frame, people):

    for i in people.keys():
        person = people[i]
        x1_b, y1_b, x2_b, y2_b = person['full_body']
        x1_h, y1_h, x2_h, y2_h = person['head']
        x1_tb, y1_tb, x2_tb, y2_tb = person['body']
        accessory = person['accessory']
        name = person['name']
        card = 0
        logo = 0
        for box, clss_id in zip(*accessory):
            x1, y1, x2, y2 = box
            if clss_id == 'card':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                card += 1
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                logo += 1

        body_color = (0, 0, 255)
        if card > 0 and logo > 0:
            body_color = (0, 255, 0)
        elif card > 0 or logo > 0:
            body_color = (0, 255, 255)
        cv2.rectangle(frame, (x1_b, y1_b), (x2_b, y2_b), body_color, 2)
        cv2.rectangle(frame, (x1_h, y1_h), (x2_h, y2_h), (255, 0, 0), 2)
        cv2.putText(frame, str(i) + f'-{name}', (x1_b, y1_b + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, str(i), (x1_h, y1_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame
