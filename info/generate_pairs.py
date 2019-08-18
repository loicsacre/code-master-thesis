
import csv
import os

from itertools import combinations
from path import Paths


def generate_pairs():
    """ Generate all the pairs of images as list of 
            (tissue, dye1, dye2, path_to_img, original_name1, original_name2, extension)
    """

    tissue2info = dict()

    with open('./info/project-info.csv', 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        next(f_csv)

        for row in f_csv:
            tissue = row[1]
            dye = row[2]
            original_name = row[6]

            if tissue not in tissue2info:
                tissue2info[tissue] = set()

            tissue2info[tissue].add((dye, original_name))

    pairs = []

    for tissue in tissue2info:

        available_items = tissue2info[tissue]

        tissue_pairs = list(combinations(
            available_items, 2 if len(available_items) > 1 else 1))

        for ((dye1, original_name1), (dye2, original_name2)) in tissue_pairs:

            dirs_in_path = os.path.join(Paths.PATH_TO_IMAGES, tissue)

            dir_scale_img = [x for x in os.listdir(
                dirs_in_path) if x != '.DS_Store'][0]

            path_to_img = os.path.join(
                Paths.PATH_TO_IMAGES, tissue, dir_scale_img)

            extension = ".jpg"

            if os.path.exists(os.path.join(path_to_img, f"{original_name1}.png")):
                extension = ".png"

            pairs.append((tissue, dye1, dye2, path_to_img,
                          original_name1, original_name2, extension))

    return pairs
