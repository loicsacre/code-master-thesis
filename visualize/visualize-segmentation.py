"""
Visualize the segmentation on all of the images in the dataset
"""

import csv
import errno
import os
import random
from argparse import ArgumentParser

from path import Paths
from utils import mkdir, visualize_segmentation


def get_args():
    """get the parameters of the program"""
    parser = ArgumentParser(prog="Program to extract features")

    parser.add_argument("--size", dest='size',
                        default=300,
                        type=int,
                        help='Size of the tiles')

    parser.add_argument("--shift", dest='shift',
                        default=75,
                        type=int,
                        help='Shift for the tiles')

    parser.add_argument("--output", dest='output',
                        default="./visualize/visualize_segmentation/",
                        help='Path of the outputs')

    return parser.parse_args()


def main():
    args = get_args()

    size = args.size
    shift = args.shift

    output_dir = os.path.join(args.output, f"{size}/{shift}")
    mkdir(output_dir)

    with open('./info/project-info.csv', 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        next(f_csv)

        for row in f_csv:
            tissue = row[1]
            dye = row[2]
            original_name = row[6]
            scale = row[7]

            path_to_image = os.path.join(
                Paths.PATH_TO_IMAGES, tissue, scale, original_name)

            extension = "jpg"
            if not os.path.exists(f"{path_to_image}.{extension}"):
                extension = "png"
            path_to_image = f"{path_to_image}.{extension}"

            output_filename = os.path.join(
                output_dir, tissue + "&" + dye + ".jpg")

            print(output_filename)
            if os.path.exists(output_filename):
                print(f"{output_filename} already exists")
                continue

            visualize_segmentation(
                path_to_image, output_filename, size, shift)


if __name__ == "__main__":
    main()
