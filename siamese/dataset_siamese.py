import os
import pickle
from time import time

import numpy as np
import torch
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from random import seed, shuffle, choice, randint

import getpass
username = getpass.getuser()

root = os.path.join("/scratch", username)
print("####")
if not os.path.isdir(root):
    print(f"# {root} does not exist")
    root = "."
else:
    print(f"# {root} exists")
print("####\n")


class CytomineDataset(Dataset):

    def __init__(self, root_path=os.path.join(root, "datasets/patches/300"), dataset_path="./datasets/cnn/dataset-networks.data", dataset_type="training", transform=None, adaptative_transform=None, loader=default_loader, batch_size=32, verbose=False):
        super(CytomineDataset, self).__init__()
        assert(os.path.exists(root_path))
        assert((transform is not None) != (adaptative_transform is not None) or (transform is None) or(adaptative_transform is None))
        
        if transform is None and adaptative_transform is None:
            # return at least a tensor
            transform = transforms.Compose([transforms.ToTensor()])

        if adaptative_transform is None:
            print("Transform..")
            for t in transform.transforms:
                print(f"  ## {t}")
        else:
            print("Adaptive transform..")

        if dataset_type == "training":
            seed(3796)
        if dataset_type == "evaluating":
            seed(2406)
        if dataset_type == "testing":
            seed(1234)
        self.dataset_type = dataset_type

        self.root = root_path
        self.loader = loader
        self.transform = transform
        self.adaptative_transform = adaptative_transform
        self.batch_size = batch_size
        self.verbose = verbose

        self.loading_time_batch = 0
        self.loading_time_batch2 = 0
        self.counter = 0

        self.cache = dict()

        self.positive_label = 1
        self.negative_label = 0

        with open(dataset_path, 'rb') as input_file:
            data = pickle.load(input_file)

        self.data = data[dataset_type]["similar"]
        self.data_set = set(self.data)
        # self.bad_patches = data["bad"]

        # transpose with PIL (https://github.com/python-pillow/Pillow/blob/master/src/PIL/Image.py)
        # FLIP_LEFT_RIGHT = 0
        # FLIP_TOP_BOTTOM = 1
        # ROTATE_90 = 2
        # ROTATE_180 = 3
        # ROTATE_270 = 4
        # TRANSPOSE = 5
        # TRANSVERSE = 6
        self.nb_of_transposes = 7

        self.length = len(self.data)*(1+self.nb_of_transposes)
        self.index_sampling_sim = list(range(self.length))
        self.index_sampling_dissim = list(range(self.length))

        shuffle(self.index_sampling_sim)
        shuffle(self.index_sampling_dissim)

        print("## {} dataset: {} triplet samples".format(
            dataset_type, self.length))

    def __len__(self):
        shuffle(self.index_sampling_sim)
        shuffle(self.index_sampling_dissim)

        return self.length

    def load_and_transform(self, path, index, tissue, dye):

        if path not in self.cache:
            sample = self.loader(path)
            self.cache[path] = sample
        else:
            sample = self.cache[path]

        if index % (self.nb_of_transposes + 1) != 0:
            transpose = index % (self.nb_of_transposes + 1) - 1
            sample = sample.transpose(transpose)

        if self.adaptative_transform is not None:
            sample = self.adaptative_transform.transform(tissue, dye)(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_negative(self, sample_info, index):

        tissue, dye, landmark_nb = sample_info

        path = os.path.join(
            self.root, tissue, dye, f"{landmark_nb}.jpg")

        sample_negative = self.load_and_transform(path, index, tissue, dye)

        return sample_negative

    def get_anchor_and_positive(self, pair, index):

        tissue, (dye1, dye2), landmark_nb = pair

        path1 = os.path.join(
            self.root, tissue, dye1, f"{landmark_nb}.jpg")

        path2 = os.path.join(
            self.root, tissue, dye2, f"{landmark_nb}.jpg")

        sample_anchor = self.load_and_transform(path1, index, tissue, dye1)
        sample_positive = self.load_and_transform(path2, index, tissue, dye2)

        return sample_anchor, sample_positive

    def get_dissimilar(self, tissue, ref_dye, landmark_nb):

        possible_dyes = set([x[1][0] for x in self.data_set if x[0]
                             == tissue and x[1][0] != ref_dye])
        possible_dyes.update([x[1][1] for x in self.data_set if x[0]
                              == tissue and x[1][1] != ref_dye])

        dye = choice(list(possible_dyes))
        highest_landmark_nb = max(
            [x[2] for x in self.data_set if x[0] == tissue])
        while True:
            i = randint(0, highest_landmark_nb)
            while i == landmark_nb:
                i = randint(0, highest_landmark_nb)
            if (tissue, (ref_dye, dye), i) in self.data_set or (tissue, (dye, ref_dye), i) in self.data_set:
                break

        return tissue, dye, i

    def __getitem__(self, index):

        start_time = time()

        i = self.index_sampling_sim[index]

        # As data contains only the pairs (not the augmented one),
        # get the true index by dividing by the number of transforms + 1
        i_original = i//(self.nb_of_transposes + 1)

        # Get the similar pair (anchor + positive)
        anchor, positive = self.get_anchor_and_positive(
            self.data[i_original], i)
    
        # Get the anchor info
        (tissue, (anchor_dye, _), anchor_landmark) = self.data[i_original]

        # get a sample with a different dye and landmark nb
        (_, negative_dye, negative_landmark_nb) = self.get_dissimilar(
            tissue, anchor_dye, anchor_landmark)
        negative = self.get_negative(
            (tissue, negative_dye, negative_landmark_nb), i)

        self.loading_time_batch += (time() - start_time)

        if self.verbose and index % (self.batch_size//2) == (self.batch_size//2-1):
            print(
                f"** Elapsed time to load batch {index//(self.batch_size//2)}: {self.loading_time_batch:3f} s")
            self.loading_time_batch = 0

        return (anchor, positive, negative)

# num_output_channels = 1
# resize = 64
# transform_composition = []
# transform_composition.append(transforms.Grayscale(num_output_channels=num_output_channels))
# transform_composition.append(transforms.Resize((resize, resize)))
# transform_composition.append(transforms.ToTensor())
# transform_composition.append(transforms.Normalize(
#     [0.5]*num_output_channels, [0.625]*num_output_channels))
# transform = transforms.Compose(transform_composition)

# dataset = CytomineDataset(dataset_type="training", transform=transform)

# batch_size = 64
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size, shuffle=False)

# for j in range(2):
#     print("--")
#     for i, (a, b, c) in enumerate(dataloader):
#         # print(c, d)
#         print(i, a.size(), b.size(), c.size())
#         #break
#     break
