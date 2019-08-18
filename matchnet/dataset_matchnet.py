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
        if dataset_type is not "training":
            self.data_dis = data[dataset_type]["dissimilar"]
        
        self.data_set = set(self.data)

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

        print("## {} dataset: {} samples".format(
            dataset_type, self.length*2))

    def __len__(self):
        shuffle(self.index_sampling_sim)
        shuffle(self.index_sampling_dissim)

        return self.length

    def load_and_transform(self, path1, path2, index, tissue, dye1, dye2):

        start_time = time()

        if path1 not in self.cache:
            sample1 = self.loader(path1)
            self.cache[path1] = sample1
        else:
            sample1 = self.cache[path1]

        if path2 not in self.cache:
            sample2 = self.loader(path2)
            self.cache[path2] = sample2
        else:
            sample2 = self.cache[path2]

        self.loading_time_batch2 += time() - start_time

        if self.verbose and (self.counter % self.batch_size) == (self.batch_size-1):
            print(
                f"*** Elapsed time pure loading batch: {self.loading_time_batch2:3f} s")
            self.loading_time_batch2 = 0
        self.counter += 1

        if index % (self.nb_of_transposes + 1) != 0:
            transpose = index % (self.nb_of_transposes + 1) - 1
            sample1 = sample1.transpose(transpose)
            sample2 = sample2.transpose(transpose)

        if self.adaptative_transform is not None:
            sample1 = self.adaptative_transform.transform(tissue, dye1)(sample1)
            sample2 = self.adaptative_transform.transform(tissue, dye2)(sample2)

        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return sample1, sample2

    def get_pair(self, pair, index, target, sim=True):

        if sim:
            tissue, (dye1, dye2), landmark_nb = pair
            landmark_nb1 = landmark_nb2 = landmark_nb
        else:
            tissue, (dye1, landmark_nb1), (dye2, landmark_nb2) = pair

        path1 = os.path.join(
            self.root, tissue, dye1, f"{landmark_nb1}.jpg")

        path2 = os.path.join(
            self.root, tissue, dye2, f"{landmark_nb2}.jpg")

        sample1, sample2 = self.load_and_transform(path1, path2, index, tissue, dye1, dye2)

        return np.stack((sample1, sample2)), torch.tensor(target).type(torch.FloatTensor)

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
        j = self.index_sampling_dissim[index]

        # As data contains only the pairs (not the augmented one),
        # get the true index by dividing by the number of transforms + 1
        i_original = i//(self.nb_of_transposes + 1)
        j_original = j//(self.nb_of_transposes + 1)

        # Get the similar pair
        pair_sim, target_sim = self.get_pair(
            self.data[i_original], i, self.positive_label)

        # Get the dissimilar pair
        if self.dataset_type == "training":
            (tissue, dye_pair, landmark_nb1) = self.data[j_original]
            dye1 = dye_pair[randint(0, 1)]  # choose randomly a dye as reference
            # get a sample with a different dye and landmark nb
            (_, dye2, landmark_nb2) = self.get_dissimilar(tissue, dye1, landmark_nb1)
            pair_dis_ = (tissue, (dye1, landmark_nb1), (dye2, landmark_nb2))
        else:
            pair_dis_ = self.data_dis[j_original]
        
        pair_dis, target_dis = self.get_pair(
            pair_dis_, j, self.negative_label, False)  # make the pair

        self.loading_time_batch += (time() - start_time)

        if self.verbose and index % (self.batch_size//2) == (self.batch_size//2-1):
            print(
                f"** Elapsed time to load batch {index//(self.batch_size//2)}: {self.loading_time_batch:3f} s")
            self.loading_time_batch = 0

        return pair_sim, pair_dis, target_sim, target_dis
