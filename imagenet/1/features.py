import csv
import os
import time as time
from itertools import combinations

import numpy as np
from PIL import Image

from path import Paths
from utils import FeaturesExtractor, Normalizer, get_patches_from_landmarks

def get_features(arch, size, pooling=True):

    print("## Starting extracting features...")
    if pooling:
        print("## Using pooling..")
    else:
        print("## Not using pooling..")

    # Declare the features  extractor
    extractor = FeaturesExtractor(arch)

    normalizer = Normalizer()

    starting = time.time()

    results_features = dict()

    with open('./info/project-info.csv', 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        next(f_csv)

        for row in f_csv:
            tissue = row[1]
            dye = row[2]
            original_name = row[6]

            if tissue not in results_features:
                results_features[tissue] = dict()
            if dye not in results_features[tissue]:
                results_features[tissue][dye] = None

            patches = get_patches_from_landmarks(
                tissue, original_name, size=size)

            nb_of_landmarks = len(patches)

            for landmark_nb, (_, _, patch) in enumerate(patches):

                normalize = normalizer.get(tissue, dye)
                extractor.set_normalize(normalize)

                img = Image.fromarray(patch)

                features = extractor.get_features_from_img(
                    img, size, pooling).cpu().numpy().astype(np.float32)

                if landmark_nb == 0:
                    results_features[tissue][dye] = np.zeros(
                        (nb_of_landmarks, features.shape[0]), dtype=np.float32)

                results_features[tissue][dye][landmark_nb] = features

    print("   Elapsed time : {}".format(time.time() - starting))

    return results_features
