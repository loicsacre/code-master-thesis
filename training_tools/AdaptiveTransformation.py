"""
AdaptiveTransformation: class which adapts the normalization according to the dye
"""

from utils import Normalizer
from torchvision import transforms


class AdaptiveTransformation():

    def __init__(self):
        self.normalizer = Normalizer()

    def transform(self, tissue, dye):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalizer.get(tissue, dye)
        ])
