
from itertools import combinations
import numpy as np
from time import time

from utils import compare as comp

def compare(features, distance="cos"):
    """ generate the similarity measures for patch-pair combinations """

    start_time = time()

    results = {}

    for tissue in features:

        dyes = list(features[tissue].keys())
        dyes_pair = list(combinations(dyes, 2 if len(dyes) > 1 else 1))

        results[tissue] = {}

        for dye1, dye2 in dyes_pair:

            nb_annot = features[tissue][dye1].shape[0]
            result_pair = np.ones((nb_annot, nb_annot)) 

            features_dye1 = features[tissue][dye1]
            features_dye2 = features[tissue][dye2]

            result_pair = comp(features_dye1, features_dye2, distance)
            results[tissue][(dye1, dye2)] = result_pair

    print(f"==> Time to compare : {time() - start_time} sec")
    return results