import cv2
import numpy as np


def read_img(img_str: str, target_size: int) -> np.ndarray:
    img = cv2.imread(img_str, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    return img


def draw_boxes(img: str, boxes:list) -> np.ndarray:
    for i in range(len(boxes)):
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]),
                      (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (0, 0, 255), 2)

    return img


def draw_grid(img: str, pixel_step: int) -> np.ndarray:
    x = pixel_step
    y = pixel_step

    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=(255, 255, 255))
        x += pixel_step

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=(255, 255, 255))
        y += pixel_step   

    return img


def draw_text(img: str, texts: list, locations: list):
    for text, loc in zip(texts, locations):
        cv2.putText(img, text, (loc[0], loc[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
    return img
