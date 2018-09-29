import json
import pandas as pd


def load_pascal(json_path: str) -> (dict, list):
    """ Return list is a list with each element is of the format:
        ['2008_000008.jpg', 500, 442, ['horse', 'person'], [[52, 86, 419, 334], [157, 43, 132, 124]], [12, 14]].
        [img name, width of image, height of image, [class names], [bounding boxes], [class id]].
    """

    json_data = json.load(open(json_path))

    cats = json_data["categories"]
    id_cat = []

    for c in cats:
        id_cat.append([c["id"], c["name"]])

    df_cats = pd.DataFrame(id_cat, columns=["category_id", "name"])

    id_cat = {value-1: key for (value, key) in id_cat}

    df_filename = pd.DataFrame(json_data["images"])
    df_filename.columns = ["file_name", "height", "image_id", "width"]

    df_bbox = pd.DataFrame(json_data["annotations"])

    df = df_filename.merge(df_bbox, on="image_id")
    df = df[df["ignore"] == 0]
    df = df.drop(["area", "ignore", "iscrowd", "segmentation", "image_id"], axis=1)
    df = df.merge(df_cats, on="category_id")

    grouped_data = []

    grouped = df.groupby("file_name")
    for name, group in grouped:
        val = [name, group["width"].values[0], group["height"].values[0], list(group["name"].values),
               list(group["bbox"].values), list(group["category_id"].values - 1)]
        grouped_data.append(val)

    return (id_cat, grouped_data)


def rescale_bounding_boxes(data_list: list, target_size: int) -> list:
    """ Return list is a list with each element is of the format:
        ['2008_000050.jpg', ['car'], [6], [[130, 117, 17, 19]]].
        [img name, [class name], [class id], [bounding boxes]].
    """

    for d in data_list:
        x_scale = target_size / d[1]
        y_scale = target_size / d[2]

        old_boxes = d[4]
        new_boxes = []

        for i in range(len(old_boxes)):
            (x, y, d_x, d_y) = old_boxes[i]

            x = int(round(x * x_scale))
            y = int(round(y * y_scale))
            d_x = int(round(d_x * x_scale))
            d_y = int(round(d_y * y_scale))

            new_boxes.append([x, y, d_x, d_y])

        d[4] = new_boxes

        # removing width and height of image
        del d[2]
        del d[1]

    # Re-orginize the list
    data_list = [[d[0], d[1], d[3], d[2]] for d in data_list]
    return data_list


def convert_to_center(data_list: list) -> list:
    """
    Converting [bx, by, w, h] to [cx, cy, w, h].
    """

    for d in data_list:
        for box in d[3]:
            box[0] = box[0] + box[2]/2
            box[1] = box[1] + box[3]/2

    return data_list
