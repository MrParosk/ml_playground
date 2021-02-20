import cv2
import matplotlib.pyplot as plt


def plot_boxes(img, boxes, names):
    color = (0, 255, 0)
    box_tickness = 2
    text_tickness = 1

    assert len(boxes) == len(names)

    for i in range(len(boxes)):
        box = boxes[i]
        name = names[i]

        min_ = (box[0], box[1])
        max_ = (box[2], box[3])
        cv2.rectangle(img, min_, max_ ,color, box_tickness)

        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        cv2.putText(img, name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, text_tickness)
    
    return img


def plot_img_seg(img, seg_img, alpha=0.5):
    return cv2.addWeighted(img, alpha, seg_img, (1.0 - alpha), 0.0)


def plot_image(img, convert_format=cv2.COLOR_BGR2RGB):
    if convert_format:
        plot_img = cv2.cvtColor(img, convert_format)
    else:
        plot_img = img

    plt.imshow(plot_img)
    plt.show()
