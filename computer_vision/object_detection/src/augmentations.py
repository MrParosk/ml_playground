import random
import numpy as np

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img_center = img.shape[0] / 2
            img =  img[:,::-1,:]
            img = np.ascontiguousarray(img)

            bboxes[:, 0] += 2*(img_center - bboxes[:,0])
        return img, bboxes

class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5, p=0.5):
        self.lower = lower
        self.upper = upper
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img
