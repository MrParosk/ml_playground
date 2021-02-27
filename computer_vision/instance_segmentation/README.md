# Instance segmentation

Implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870).

## Data

The model is trained on PASCAL VOC 2012, which can be downloaded [here](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).

The data needs to be converted, which can be done by running (make sure you are in the root folder of this project):

```bash
python3 src/convert_annotations.py
```
