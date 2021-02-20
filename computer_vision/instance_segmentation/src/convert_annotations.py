import json
import os
import glob
import xmltodict

from classes import classes_idx


def convert_annotations(path):
    anno_files = glob.glob(os.path.join(path, "Annotations", "*.xml"))

    annos = []
    for anno_file in anno_files:
        with open(anno_file, "r") as fp:
            txt = fp.read()
            anno = xmltodict.parse(txt)

        segmented = bool(int(anno["annotation"]["segmented"]))

        if not segmented:
            continue

        id_ = anno["annotation"]["filename"].split(".")[0]

        new_anno = {
            "file_name": anno["annotation"]["filename"],
            "seg_file_name": id_ + ".png",
            "id": id_,
            ""
            "height": anno["annotation"]["size"]["height"],
            "width": anno["annotation"]["size"]["width"],
            "annotations": []
        }

        try:
            for obj in anno["annotation"]["object"]:
                # Boxes are xmin, ymin, xmax, ymax

                d = {
                    "class_name": obj["name"],
                    "class_idx": classes_idx[obj["name"]],
                    "bbox": [int(obj["bndbox"]["xmin"]),
                             int(obj["bndbox"]["ymin"]),
                             int(obj["bndbox"]["xmax"]),
                             int(obj["bndbox"]["ymax"]),
                            ]
                }

                new_anno["annotations"].append(d)
            annos.append(new_anno)
        except Exception as e:
            pass
    return annos


if __name__ == "__main__":
    PATH = "data/VOCdevkit/VOC2012"
    annos = convert_annotations(PATH)

    with open("data/annotations.json", "w") as fp:
        json.dump(annos, fp)
