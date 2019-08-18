import os
import pickle
import time

import numpy as np
import torch
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from random import seed, shuffle

from time import time


import getpass
username = getpass.getuser()

root = os.path.join("scratch", username)
if not os.path.isdir(os.path.join("scratch", username)):
    root = "."


seed(3796)

class CytomineDataset(Dataset):

    def __init__(self, root=os.path.join(root, "datasets/patches/300"), dataset_path="./datasets/cnn/dataset-networks.data", dataset_type="training", transform=None, loader=default_loader, batch_size=32, verbose=False):
        super(CytomineDataset, self).__init__()
        assert os.path.exists(root)

        self.loader = loader
        self.transform = transform
        self.batch_size = batch_size
        self.verbose = verbose

        self.positive_label = 1
        self.negative_label = 0

        self.loading_time_batch = 0
        self.loading_time_batch2 = 0
        self.counter = 0

        self.cache = dict()

        with open(dataset_path, 'rb') as input_file:
            dataset = pickle.load(input_file)

        dataset = dataset[dataset_type]

        len_data_similar = len(dataset["similar"])
        len_data_dissimilar = len(dataset["dissimilar"])
        total_nb_data = len_data_similar + len_data_dissimilar

        # transpose with PIL (https://github.com/python-pillow/Pillow/blob/master/src/PIL/Image.py)
        # FLIP_LEFT_RIGHT = 0
        # FLIP_TOP_BOTTOM = 1
        # ROTATE_90 = 2
        # ROTATE_180 = 3
        # ROTATE_270 = 4
        # TRANSPOSE = 5
        # TRANSVERSE = 6
        self.nb_of_transposes = 7

        self.data_sim = []
        self.data_dis = []

        for i, (sim, dissim) in enumerate(zip(dataset["similar"], dataset["dissimilar"])):

            tissue, (dye1, dye2), landmark_nb = sim

            self.data_sim.append((
                os.path.join(
                    root, tissue, dye1, f"{landmark_nb}.jpg"),
                os.path.join(
                    root, tissue, dye2, f"{landmark_nb}.jpg"), self.positive_label))

            (tissue1, dye1, landmark_nb1),  (tissue2, dye2, landmark_nb2) = dissim

            self.data_dis.append((
                os.path.join(
                    root, tissue1, dye1, f"{landmark_nb1}.jpg"),
                os.path.join(
                    root, tissue2, dye2, f"{landmark_nb2}.jpg"), self.negative_label))

        self.length = len(self.data_sim)*(1+self.nb_of_transposes)
        self.index_sampling_sim = list(range(self.length))
        self.index_sampling_dissim = list(range(self.length))

        print("## {} dataset: {} samples".format(
            dataset_type, self.length*2))

    def __len__(self):
        shuffle(self.index_sampling_sim)
        shuffle(self.index_sampling_dissim)
        return self.length

    def get_pair(self, pair, index):
        path1, path2, target = pair

        start_time = time()
            
        # if path1 not in self.cache:
        #     sample1 = self.loader(path1)
        #     self.cache[path1] = sample1
        # else:
        #     sample1 = self.cache[path1]

        # if path2 not in self.cache:
        #     sample2 = self.loader(path2)
        #     self.cache[path2] = sample2
        # else:
        #     sample2 = self.cache[path2]

        sample1 = self.loader(path1)
        sample2 = self.loader(path2)

        self.loading_time_batch2 += time() - start_time

        if self.verbose and (self.counter % self.batch_size) == (self.batch_size-1):
            print(f"*** Elapsed time pure loading batch: {self.loading_time_batch2:3f} s")
            self.loading_time_batch2 = 0
        self.counter += 1

        if index % (self.nb_of_transposes + 1) != 0:
            transpose = index % (self.nb_of_transposes + 1) - 1
            sample1 = sample1.transpose(transpose)
            sample2 = sample2.transpose(transpose)

        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return np.stack((sample1, sample2)), target

    def __getitem__(self, index):

        start_time = time()

        i = self.index_sampling_sim[index]
        j = self.index_sampling_dissim[index]

        # As data contains only the pairs (not the augmented one), 
        # get the true index by dividing by the number of transforms + 1
        i_original = i//(self.nb_of_transposes + 1)
        j_original = j//(self.nb_of_transposes + 1)

        pair_sim, target_sim = self.get_pair(self.data_sim[i_original], i)
        pair_dis, target_dis = self.get_pair(self.data_dis[j_original], j)

        self.loading_time_batch += (time() - start_time)
        
        if self.verbose and index % (self.batch_size//2) == (self.batch_size//2 - 1):
            print(f"** Elapsed time to load batch {index//16}: {self.loading_time_batch:3f} s")
            self.loading_time_batch = 0

        return pair_sim, pair_dis, \
            torch.tensor(self.positive_label).type(torch.FloatTensor), \
            torch.tensor(self.negative_label).type(torch.FloatTensor)



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

# batch_size = 32
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size//2, shuffle=False)

# for j in range(2):
#     print("--")
#     for i, (a, b, c, d) in enumerate(dataloader):
#         print(c, d)
#         print(i, a.size(), b.size(), c.size(), d.size())
#         #break
#     break


# img = Image.open("./datasets/patches/100/mice-kidney_1/PAS/0.jpg")

# np_array = np.array([[255, 255], [0, 0]], dtype=np.uint8)
# img = Image.fromarray(np_array)
# num_output_channels = 3
# transform_composition = []
# transform_composition.append(transforms.Grayscale(num_output_channels=num_output_channels))
# transform_composition.append(transforms.ToTensor())
# transform_composition.append(transforms.Normalize(
#     [0.5]*num_output_channels, [0.625]*num_output_channels))
# transform = transforms.Compose(transform_composition)

# print(transform(img))
# # print(np.array(img.convert("L")))

# # data = []
# # data.append([transform(img)*0.8, transform(img)*0.8])
# # print(data)

# # print((np.array(transform(img))))
