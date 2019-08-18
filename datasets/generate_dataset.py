"""
Generate the dataset partitions (training, evaluating, testing) for the training
"""

import csv
import os
import pickle
import time
from argparse import ArgumentParser
from itertools import combinations
from math import factorial
from random import choice, randint, seed, shuffle

import numpy as np
from PIL import Image
from tqdm import tqdm

from path import Paths
from utils import mkdir, generate_dataset

seed(3796)


"""

The dataset is a dict with keys:

'training' : training dataset
'evaluating' : evaluating dataset
'testing' : testing dataset

which are themselves dict with key

'similar' for patches which are similar (list of tuples (tissue, (dye1, dye2), landmark))
'dissimilar' for patches which are dissimilar (list of tuples ((tissue1, dye1, landmark1),(tissue2, dye2, landmark2)))

"""
# The size of the patches to train the networks (be sure that datasets/{size} exists!)
SIZE = 300

# Proportion for the different sets
TRAIN = 0.65
EVAL = 0.15
TEST = 0.2


class STOP(Exception):
    pass


def generate_dissimilar(data):

    # choose randomnly one of the dye
    samples = list(
        map(lambda x: [x[0], x[1][randint(0, 1)], x[2]], data))

    possible_target_samples = []
    for tissue, (dye1, dye2), landmark_nb in data:
        possible_target_samples.append((tissue, dye1, landmark_nb))
        possible_target_samples.append((tissue, dye2, landmark_nb))

    list_dissimilar = []

    shuffle(samples)

    for i, sample in tqdm(enumerate(samples)):

        tissue_ref = sample[0]
        dye_ref = sample[1]
        landmark_ref = sample[2]

        while True:

            target_sample = choice(possible_target_samples)
            tissue_target = target_sample[0]
            dye_target = target_sample[1]
            landmark_target = target_sample[2]

            # same tissue, different dyes and different landmark number
            if tissue_target == tissue_ref and dye_target != dye_ref and landmark_target != landmark_ref:
                possible_target_samples.remove(target_sample)
                break

        list_dissimilar.append(
            (tissue_ref, (dye_ref, landmark_ref), (dye_target, landmark_target)))

    return list_dissimilar


def get_info():

    data_info = dict()  # key :tissue, value tuple (nb of landmarks, list of available dyes)

    with open('./info/project-info.csv', 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        next(f_csv)

        for row in f_csv:
            tissue = row[1]
            dye = row[2]
            nb_of_landmarks = row[5]

            if tissue not in data_info:
                data_info[tissue] = (int(nb_of_landmarks), [])
            data_info[tissue][1].append(dye)

    return data_info


def get_bad_patches(data_info, size=300):
    """ Get the patches which do not have the correct dimensions """

    bad_patches = dict()

    print("Bad patches...")
    bad_patches = set()

    for tissue in data_info:
        for dye in data_info[tissue][1]:
            for i in range(data_info[tissue][0]):
                path = os.path.join(Paths.PATH_TO_PATCHES,
                                    size, tissue, dye, f"{i}.jpg")
                with Image.open(path) as image:
                    if image.size[0] != size or image.size[1] != size:
                        print(f"{path} : size {image.size}")
                        bad_patches.add((tissue, dye, i))

    return bad_patches


def main():
    """Main function"""

    # dict -> key :tissue, value tuple (nb of landmarks, list of available dyes)
    data_info = get_info()
    # list of tuples (tissue, dye, landmark_nb)
    bad_patches = get_bad_patches(data_info, size=SIZE)

    tissue2nb_landmark_pairs = {}
    counter_pairs = 0
    counter_pairs_annotation = 0
    data = []

    for tissue in data_info:

        dyes = data_info[tissue][1]
        n = len(dyes)
        r = 2
        nb_pairs = factorial(n)//(factorial(r)*factorial(n-r))

        counter_pairs += nb_pairs

        nb_of_landmarks = data_info[tissue][0]

        counter_pairs_annotation += (nb_pairs * nb_of_landmarks)

        tissue2nb_landmark_pairs[tissue] = (nb_pairs * nb_of_landmarks)

        dye_pairs = list(combinations(dyes, 2 if len(dyes) > 1 else 1))

        for pair in dye_pairs:
            for i in range(nb_of_landmarks):

                if (tissue, pair[0], i) in bad_patches or (tissue, pair[1], i) in bad_patches:
                    continue
                data.append((tissue, pair, i))

    print("")
    print(f"Number of pairs : {counter_pairs}")
    print(
        f"Number of pairs of similar landmarks (with/without bad patches): {counter_pairs_annotation}/{len(data)}")
    print(
        f"Expected number of pairs of similar landmarks per dataset: {len(data)} ({TRAIN*100}% : {len(data)*TRAIN} | {EVAL*100}% : {len(data)*EVAL} : | {TEST*100}% : {len(data)*TEST})")

    tissue2nb_landmark_pairs = sorted(
        tissue2nb_landmark_pairs.items(), key=lambda kv: kv[1], reverse=True)

    name_datasets = ["training", "evaluating", "testing"]
    tissue_per_dataset = dict((k, []) for k in name_datasets)

    c_train, c_eval, c_test = 0, 0, 0
    for i, (k, v) in enumerate(tissue2nb_landmark_pairs):

        if i % 3 == 0 and c_train + v <= len(data)*(TRAIN):
            tissue_per_dataset[name_datasets[0]].append(k)
            c_train += v
        elif i % 3 == 1 and c_eval + v <= len(data)*(EVAL):
            tissue_per_dataset[name_datasets[1]].append(k)
            c_eval += v
        elif i % 3 == 2 and c_test + v <= len(data)*(TEST):
            tissue_per_dataset[name_datasets[2]].append(k)
            c_test += v
        else:
            tissue_per_dataset[name_datasets[0]].append(k)

    dataset = dict()

    for name in name_datasets:
        data_similar = list(
            filter(lambda x: x[0] in tissue_per_dataset[name], data))
        data_dissimilar = generate_dissimilar(data_similar)

        dataset[name] = {"similar": data_similar,
                         "dissimilar": data_dissimilar}

    # Summary

    print("")
    print("Tissues for the different datasets:")
    for name in name_datasets:
        print(name, len(tissue_per_dataset[name]), tissue_per_dataset[name])

    print("")
    total = 0
    for name in name_datasets:
        nb_of_similar = len(dataset[name]["similar"])
        nb_of_dissimilar = len(dataset[name]["dissimilar"])

        nb_pair_dyes = len(set([(x[0], x[1])
                                for x in dataset[name]["similar"]]))

        total += nb_of_similar + nb_of_dissimilar
        print(f"{name} data (sim/dissim) : {nb_of_similar}/{nb_of_dissimilar}")
        print(f"Number pairs of dyes : {nb_pair_dyes}")

    print(f"=> Total : {total}")

    with open("./datasets/cnn/dataset-networks.data", 'wb') as input_file:
        pickle.dump(dataset, input_file)


if __name__ == "__main__":
    main()
