import json
import pandas as pd
from collections import namedtuple
import torch

ImageEntry = namedtuple("ImageEntry", ["filename", "width", "height",
                                       "classnames", "class_id",
                                       "bounding_boxes"
                                       ])


def load_pascal(json_path):
    json_data = json.load(open(json_path))

    images_df = pd.DataFrame(json_data["images"])
    anno_df = pd.DataFrame(json_data["annotations"])

    anno_df = anno_df[["image_id", "bbox", "category_id"]]
    anno_df = anno_df.rename(columns={"image_id": "id"})

    id_classname = {}
    for row in json_data["categories"]:
        id_classname[row["id"]] = row["name"]

    anno_df["classname"] = anno_df.apply(lambda x: id_classname[x["category_id"]], axis=1)
    df = anno_df.merge(images_df, on="id")

    grouped_data = []
    grouped = df.groupby("file_name")
    for name, group in grouped:
        val = ImageEntry(filename=name, width=group["width"].values[0], height=group["height"].values[0],
                         classnames=list(group["classname"].values), class_id=list(group["category_id"].values - 1),
                         bounding_boxes=list(group["bbox"].values))
        grouped_data.append(val)
    return id_classname, grouped_data


def rescale_bounding_boxes(data_list, target_size):
    """
    Rescaling the bounding boxes according to the new image size (target_size).
    """

    for i in range(len(data_list)):
        d = data_list[i]
        x_scale = target_size / d.width
        y_scale = target_size / d.height

        new_boxes = []
        for box in d.bounding_boxes:
            (x, y, d_x, d_y) = box

            x = int(round(x * x_scale))
            y = int(round(y * y_scale))
            d_x = int(round(d_x * x_scale))
            d_y = int(round(d_y * y_scale))

            new_boxes.append([x, y, d_x, d_y])

        data_list[i] = data_list[i]._replace(bounding_boxes=new_boxes)
    return data_list


def convert_to_center(data_list):
    """
    Converting [bx, by, w, h] to [cx, cy, w, h].
    """

    for i in range(len(data_list)):
        d = data_list[i]

        new_boxes = []
        for box in d.bounding_boxes:
            cx = box[0] + box[2]/2
            cy = box[1] + box[3]/2
            new_boxes.append([cx, cy, box[2], box[3]])
        data_list[i] = data_list[i]._replace(bounding_boxes=new_boxes)
    return data_list


def invert_transformation(bb_hat, anchors):
    """
    Invert the transform from "loc_transformation".
    """

    return torch.stack([anchors[:, 0] + bb_hat[:, 0] * anchors[:, 2],
                        anchors[:, 1] + bb_hat[:, 1] * anchors[:, 3],
                        anchors[:, 2] * torch.exp(bb_hat[:, 2]),
                        anchors[:, 3] * torch.exp(bb_hat[:, 3])
                        ], dim=1)
