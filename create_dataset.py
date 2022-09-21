# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/20/2022


import tqdm
from detector import human
import glob
from utils.create_data_utils import *

human_detector = human.Detector(cuda=False)

card = glob.glob('object/card*')
logo = glob.glob('object/logo*')


def creater(images, save_dir, step):
    c = 0
    for i in tqdm.tqdm(images):
        frame = cv2.imread(i)
        ori = frame.copy()
        for i, person in human_detector.detect(frame).items():
            if len(person) == 2:
                try:
                    body, head = person
                    # body crop
                    x1_h, y1_h, x2_h, y2_h = head
                    x1_b, y1_b, x2_b, y2_b = body
                    y_chin = y2_h
                    center_head = (x1_h + (x2_h - x1_h) // 2, y1_h + (y2_h - y1_h) // 2)
                    h_top_body = int((y2_h - y1_h) * 2)
                    w_shoulder = x2_h - x1_h
                    x1_tb, y1_tb, x2_tb, y2_tb = (
                        center_head[0] - w_shoulder, y_chin, center_head[0] + w_shoulder, y_chin + h_top_body)
                    cv2.rectangle(frame, (x1_tb, y1_tb), (x2_tb, y2_tb), (0, 0, 255), 2)
                    body = ori[y1_tb:y2_tb, x1_tb:x2_tb, :].copy()
                    h_body, w_body = body.shape[:2]
                    if h_body < 150 or w_body<150:
                        continue
                    if h_body / w_body<0.5:
                        continue
                    # stage 1: card
                    w_card, h_card = (int(0.2*w_body), int(0.3*w_body)), (int(0.25*h_body), int(0.35*h_body))
                    x_center, y_center = (int(0.2*w_body), int(0.8*w_body)), (int(0.5*h_body), int(0.8*h_body))
                    card_poly = random_parallelogram(w_card, h_card, x_center, y_center)
                    obj = cv2.imread(card[random.randint(0, len(card) - 1)], cv2.IMREAD_UNCHANGED)
                    mask = obj[:, :, 3]
                    obj = augment(obj)
                    obj[:, :, 3] = mask
                    result = add_obj_to_bg(body, obj, card_poly, brightness_range=[-10, 100])

                    # stage 2: logo
                    w_card, h_card = (int(0.15*w_body), int(0.25*w_body)), (int(0.03*h_body), int(0.1*h_body))
                    x_center, y_center = (int(0.6*w_body), int(0.8*w_body)), (int(0.1*h_body), int(0.3*h_body))
                    logo_poly = random_parallelogram(w_card, h_card, x_center, y_center)
                    obj = cv2.imread(logo[random.randint(0, len(logo) - 1)], cv2.IMREAD_UNCHANGED)
                    mask = obj[:, :, 3]
                    obj = augment(obj)
                    obj[:, :, 3] = mask
                    # if random.randint(0, 0):
                    #     result = add_obj_to_bg(result, obj, logo_poly, random_obj=True)
                    # else:
                    result = add_obj_to_bg(body, obj, logo_poly, brightness_range=[-10, 100])

                    save_yolo([0, 1], [card_poly, logo_poly], result, save_dir, file_name=f'img_{step}_%06d' % (c))
                    c += 1
                    # cv2.imshow('cc', result)
                    # if cv2.waitKey() & 0xff == ord('q'):
                    #     break
                except:
                    pass


if __name__ == '__main__':
    import os
    from multiprocessing import Process

    images = glob.glob(r'D:\Dataset\Crowd\*.jpg')
    length_data = len(images)
    st = 0
    step = 0.2
    list_thread = []
    while st < 1:
        step_batch = images[int(st * length_data):int((st + step) * length_data)]
        st += step
        save_dir = f'datasets/Step_%.2f' % (st)
        try:
            os.mkdir(save_dir)
        except:
            pass
        a = Process(target=creater, args=[step_batch, save_dir, st])
        list_thread.append(a)
    for i in list_thread:
        i.start()
