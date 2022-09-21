# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/20/2022
import cv2
import numpy as np
import random
from .color import color
import math
from imgaug import augmenters as iaa
import imgaug as ia


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def four_point_transform(img, polygon):
    h, w, chanel = img.shape
    pts = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype="float32")
    dst = np.array(polygon, dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    # warped = cv2.blur(warped, ksize=(3, 3))
    return warped


def change_brightness(img, value=30):
    '''Truyen vao img, value <0: giam do sang, value >0 : tang do sang '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def add_obj_to_bg(bg, object_4_channel, polygon, brightness_range=[-30, 100], random_obj=False):
    h, w = bg.shape[:2]
    object_4_channel = cv2.resize(object_4_channel, (w, h))
    object_4_channel = four_point_transform(object_4_channel, polygon)
    img = object_4_channel[:, :, :3]
    img = change_brightness(img, random.randint(*brightness_range))
    mask = object_4_channel[:, :, 3]
    idx = (mask != 0)

    area = bg[h - 20:h + 20, w - 20:w + 20, :]
    avg_color_per_row = np.average(area, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    if random_obj:
        rand_color = color[random.randint(0, len(color) - 1)]
        while np.linalg.norm(avg_color - rand_color) < 100:
            rand_color = color[random.randint(0, len(color) - 1)]
        img[idx] = rand_color
        noise_typ = ['gauss', 's&p', 'poisson', 'speckle']
        random.shuffle(noise_typ)
        img = noisy(noise_typ[0], img)
    bg[idx] = img[idx]
    bg = cv2.blur(bg, (3, 3))
    return bg


def rotate_around_point_lowperf(point, radians, origin=(0, 0)):
    """Rotate a point around a given point.

    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)
    return qx, qy


def random_parallelogram(w=(50, 60), h=(100, 120),
                         x_center=(200, 250),
                         y_center=(200, 250), angle=(-90, 90),
                         random_translation=(-10, 10)):
    w = random.randint(*w)
    h = random.randint(*h)
    x_center = random.randint(*x_center)
    y_center = random.randint(*y_center)
    angle = random.randint(*angle) / 180
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)

    random_translation = random.randint(*random_translation)

    x2 = x1 + w
    y2 = y1

    x3 = x2
    y3 = y1 + h

    x4 = x1
    y4 = y3

    rect = np.array([[x1 + random_translation, y1], [x2 + random_translation, y2], [x3, y3], [x4, y4]])
    poly = []
    for i in rect:
        poly.append(rotate_around_point_lowperf(i, angle, origin=(x_center, y_center)))
    return poly


def get_data_yolo(w, h, polygon, c):
    x = [polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0]]
    y = [polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1]]

    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    width = (x_max-x_min)
    height = (y_max-y_min)
    x_center = (x_min + width/2)/w
    y_center = (y_min + height/2)/h
    width = width/w
    height = height/h
    label = '%d %f %f %f %f' %(c, x_center, y_center, width, height)
    return label


def save_yolo(labels, polygons, image, save_path, file_name):
    h, w = image.shape[:2]
    data = ''
    for label, poly in zip(labels, polygons):
        data = data + '\n' + get_data_yolo(w, h, poly, label)
    data = data.strip('\n')
    with open(save_path + f'/{file_name}.txt', 'w', encoding='utf8') as f:
        f.writelines(data)
    cv2.imwrite(save_path + f'/{file_name}.jpg', image)


def distortion(img):
    A = img.shape[0] / 5
    w = 1 / img.shape[1]
    freq_x = random.random() * 3.0
    freq_y = random.random() * 1.0
    rd_x = random.randint(0, 1)
    rd_y = random.randint(0, 1)
    shift_x = lambda x: A * np.cos(freq_x * np.pi * x * w) if rd_x else A * np.sin(freq_x * np.pi * x * w)
    shift_y = lambda y: A * np.cos(freq_y * np.pi * y * w) if rd_y else A * np.sin(freq_y * np.pi * y * w)
    for i in range(img.shape[1]):
        img[:, i] = np.roll(img[:, i], int(shift_x(i)))
    for i in range(img.shape[0]):
        img[i, :] = np.roll(img[i, :], int(shift_y(i)))
    return img




def augment(image):
    #  image: bgr opencv
    augmentation_method = [
        # iaa.Affine(rotate=(-2, 2)),
        iaa.AdditiveGaussianNoise(scale=(1, 30)),
        iaa.AdditiveLaplaceNoise(),
        iaa.AdditivePoissonNoise(),
        iaa.AddToHueAndSaturation((-60, 60)),
        iaa.ElasticTransformation(alpha=50, sigma=9),  # hiệu ứng nước
        # iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0)),
        # iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Add(100), background=iaa.Multiply(0.2)),
        # iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13)),
        # iaa.BlendAlphaMask(iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()), iaa.Rain()),  # hiệu ứng mưa
        # iaa.BlendAlphaMask(iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()), iaa.Clouds()),  # hiệu ứng mây
        # iaa.BlendAlphaMask(iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()), iaa.Fog()),  # sương mù
        # iaa.BlendAlphaMask(iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()), iaa.Snowflakes()),  # tuyết
        # iaa.BlendAlphaMask(iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()), iaa.FastSnowyLandscape()),
        # iaa.MultiplyAndAddToBrightness(),
        # iaa.AddToSaturation(),
        # iaa.AddToBrightness(),
        # iaa.MultiplyBrightness(),
    ]
    seq = iaa.Sequential(random.choices(augmentation_method, k=random.randint(0, 2)))

    images = [image]
    images_aug = seq(images=images)
    return images_aug[0]
