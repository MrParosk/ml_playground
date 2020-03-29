import cv2
import numpy as np


def read_img(img_str, target_size):
    img = cv2.imread(img_str, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    return img


def draw_boxes(img, boxes):
    for box in boxes:
        cv2.rectangle(img, (int(box[0] - box[2]/2), int(box[1] - box[3]/2)),
                      (int(box[0] + box[2]/2), int(box[1] + box[3]/2)),
                      (0, 0, 255), 2)

    return img


def draw_grid(img, pixel_step):
    x = pixel_step
    y = pixel_step

    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=(255, 255, 255))
        x += pixel_step

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=(255, 255, 255))
        y += pixel_step

    return img


def draw_text(img, texts, locations):
    for text, loc in zip(texts, locations):
        cv2.putText(img, text, (int(loc[0]), int(loc[1])), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (255, 0, 0), 1)
    return img
