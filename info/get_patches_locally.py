
import csv
import os
from argparse import ArgumentParser

from PIL import Image

from path import Paths
from utils import get_patches_from_landmarks, mkdir


def get_args():
    """get the parameters of the program"""
    parser = ArgumentParser(prog="Program to get all the patches for a specific size locally")

    parser.add_argument('--output', dest='output',
                        default=Paths.PATH_TO_PATCHES,
                        help="Where the patches will be stored")

    parser.add_argument("--size", dest='size',
                        default=300,
                        type=int,
                        help='Size of the patches (in pixels)')

    return parser.parse_args()


def main():

    args = get_args()

    with open('./info/project-info.csv', 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        next(f_csv)

        for row in f_csv:

            tissue = row[1]
            dye = row[2]
            original_name = row[6]

            patches = get_patches_from_landmarks(tissue, original_name, size=args.size)

            output_dir = os.path.join(args.output, str(args.size), tissue, dye)
            mkdir(output_dir)

            for i, (_, _, patch) in enumerate(patches):

                output_file = os.path.join(output_dir, f"{i}.jpg")

                if os.path.exists(output_file):
                    continue

                with Image.fromarray(patch) as patch:
                    patch.save(output_file)

if __name__ == "__main__":
    main()
