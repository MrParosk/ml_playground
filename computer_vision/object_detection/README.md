# Object detection

Implementation of three object detection methods: 

- [YOLO (v1)](https://arxiv.org/abs/1506.02640).
- [SSD](https://arxiv.org/abs/1512.02325) (with [Focal loss](https://arxiv.org/abs/1708.02002)).
- [RetinaNet](https://arxiv.org/abs/1708.02002).
- [Faster-RCNN](https://arxiv.org/abs/1506.01497) (TBC).

They are trained on PASCAL VOC 2012, which can be found [here](https://pjreddie.com/projects/pascal-voc-dataset-mirror/). 

The notebooks uses json annotations and therefore the xml-files needs to be converted to json-format. This can be done with xml2json.py, i.e. 

```bash
python src/xml2json.py ./data/VOCdevkit 2012
```
