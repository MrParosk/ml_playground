import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_boxes(img, boxes, names, color=(0, 255, 0)):
    plot_img = np.copy(img)
    
    box_tickness = 2
    text_tickness = 1

    assert len(boxes) == len(names)

    for i in range(len(boxes)):
        box = boxes[i]
        name = names[i]

        min_ = (box[0], box[1])
        max_ = (box[2], box[3])

        cv2.rectangle(plot_img, min_, max_ , color, box_tickness)

        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        cv2.putText(plot_img, name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, text_tickness)
    
    return plot_img


def plot_img_seg(img, seg_img, mask_img, alpha=0.5):
    # mask_img specifying where the segmentation is
    seg_img_part = mask_img * cv2.addWeighted(img, alpha, seg_img, (1.0 - alpha), 0.0)
    orig_img_part = (1 - mask_img) * img
    return orig_img_part + seg_img_part


def plot_boxes_instance_masks(plot_img, seg_img, bboxes, names, seg_ids, alpha=0.5):
    for i in range(len(bboxes)):
        range_ = (0, 255)
        color = (random.randint(*range_),
                 random.randint(*range_),
                 random.randint(*range_)
                )

        plot_img = plot_boxes(plot_img, [bboxes[i]], [names[i]], color=color)

        id_ = seg_ids[i]
        mask_img = (seg_img == id_).astype(np.uint8)

        seg_img_id = mask_img * np.array(color, dtype=np.uint8).reshape((1, 1, 3))
        plot_img = plot_img_seg(plot_img, seg_img_id, mask_img)

    return plot_img


def plot_image(img, convert_format=cv2.COLOR_BGR2RGB):
    if convert_format:
        plot_img = cv2.cvtColor(img, convert_format)
    else:
        plot_img = img

    plt.imshow(plot_img)
    plt.show()
