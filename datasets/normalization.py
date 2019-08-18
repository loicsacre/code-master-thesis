"""
Generate the normalization parameters per image
"""

import csv
import os
import pickle

import numpy as np
import torchvision.transforms as transforms

from utils import segment_image, get_patches_from_landmarks
from PIL import Image

# http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                m*n/(m+n)**2 * (tmp - newmean)**2
            self.std = np.sqrt(self.std)

            self.nobservations += n


def main():
    dye2tissue = dict()

    with open('./info/project-info.csv', 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        next(f_csv)

        for row in f_csv:

            tissue = row[1]
            dye = row[2]
            original_name = row[6]
            scale = row[7]

            if dye not in dye2tissue:
                # [sum pixels channel 0, sum pixels channel 1, sum pixels channel 2]
                dye2tissue[(tissue, dye)] = (
                    StatsRecorder(), StatsRecorder(), StatsRecorder())

            path = os.path.join("./datasets/images",
                                tissue, scale, original_name)
            extension = ".jpg"
            if not os.path.exists(path + extension):
                extension = ".png"
            path += extension

            patches = segment_image(path, size=300, shift=150)

            print(tissue, dye)

            for (_, _, patch) in patches:
                
                for c in range(3):

                    patch = transforms.ToTensor()(
                        patch).numpy()  # normalize to [0,1]

                    patch = np.moveaxis(patch, 0, 2)

                    array = patch[:, :, c].reshape((-1, 1))
                    dye2tissue[(tissue, dye)][c].update(array)


    normalization_dic = dict()
    for (tissue, dye) in dye2tissue:
        print((tissue, dye))
        mean = [dye2tissue[(tissue, dye)][c].mean[0] for c in range(3)]
        std = [dye2tissue[(tissue, dye)][c].std[0] for c in range(3)]

        print(f"Dye {(tissue, dye)}: mean {mean} | std {std}")
        normalization_dic[(tissue, dye)] = (mean, std)

    print(normalization_dic)

    output_filename = "./datasets/normalization-dye-seg.info"
    with open(output_filename, 'wb') as input_file:
        pickle.dump(normalization_dic, input_file)


if __name__ == "__main__":
    main()
