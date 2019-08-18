import torchvision.transforms as transforms
import pickle


class Normalizer():
    """Get the normalization for a certain dye (see ./datasets/normalization.py to generate the data)"""

    def __init__(self, path="./datasets/normalization-dye-seg.info"):
        print("### Normalizing using the whole images (by images)")
        with open(path, 'rb') as file_:
            self.data = pickle.load(file_)

    def get(self, tissue, dye):
        mean, std = self.data[(tissue, dye)]
        return transforms.Normalize(mean=mean, std=std)
