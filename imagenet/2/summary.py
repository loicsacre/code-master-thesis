"""
Generate the summary csv's for the expermiment with imangenet 2

"""

import os
import pickle
import csv
from argparse import ArgumentParser
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from utils import mkdir


def get_args():
    """get the parameters of the program"""
    parser = ArgumentParser(prog="Program to summarize to results imagenet2")

    parser.add_argument("--size", dest='size',
                        default=300,
                        type=int,
                        help='Size of the crops')

    parser.add_argument("--distance", dest='distance',
                        default='cos',
                        choices=["cos", "eucl"],
                        help='Distance metric : \
                        cosine similarity (cos) | \
                        euclidean distance (eucl) \
                        (default: cos)')

    return parser.parse_args()


def main():

    args = get_args()
    size = args.size
    distance = args.distance

    mkdir(f'./results/imagenet/2/csv/{args.distance}/{size}/')

    for root, dirs, files in os.walk(f"./results/imagenet/2/data/{distance}/{size}/", topdown=False):

        if dirs == []:

            arch = root.split("/")[-1]

            total_nb_of_landmarks = 0
            total_nb_of_tiles = 0
            total_counter_top1 = 0
            total_counter_top5 = 0
            list_top1accuracy = []
            list_top5accuracy = []

            total_time = 0
            total_time_get_patches2 = 0
            total_time_get_features2 = 0
            total_time_comparison = 0

            with open(f'./results/imagenet/2/csv/{args.distance}/{size}/{arch}.csv', 'w') as csv_file:

                fieldnames = ['tissue', 'dye1', 'dye2', 'Nb of landmarks (dye 1)', 'Nb of tiles (dye 2)',
                              'top-1-accuracy', 'top-5-accuracy', 'time (h:m:s)', 'Time to get patches (dye2)', 'Features extraction time (dye2)', 'Comparison time']

                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                print(f"Architecture: {arch} (size {size})")

                for f in files:

                    with open(os.path.join(root, f), 'rb') as data_file:
                        data = pickle.load(data_file)

                    tissue, dye1, dye2, _, _, _, _ = data["pair"]
                    top_accuracy_list = data["top_accuracy_list"]
                    top1accuracy = top_accuracy_list[0][1]
                    top5accuracy = top_accuracy_list[1][1]
                    list_top1accuracy.append(top1accuracy)
                    list_top5accuracy.append(top5accuracy)

                    time = data["time"]
                    total_time += time

                    time_get_patches2 = data["time_get_patches2"]
                    time_get_patches2 = time_get_patches2 if time_get_patches2 != "None" else -1
                    total_time_get_patches2 += time_get_patches2
                    time_get_features2 = data["time_get_features2"]
                    time_get_features2 = time_get_features2 if time_get_features2 != "None" else -1
                    total_time_get_features2 += time_get_features2
                    time_comparison = data["time_comparison"]
                    total_time_comparison += time_comparison

                    counter = data["counter"]
                    total_counter_top1 += counter[0]
                    total_counter_top5 += counter[1]

                    results_comparison = data["results_comparison"]

                    nb_of_landmarks = results_comparison.shape[0]
                    nb_of_tiles = results_comparison.shape[1]

                    total_nb_of_landmarks += nb_of_landmarks
                    total_nb_of_tiles += nb_of_tiles

                    writer.writerow({fieldnames[0]: tissue,
                                     fieldnames[1]: dye1,
                                     fieldnames[2]: dye2,
                                     fieldnames[3]: nb_of_landmarks,
                                     fieldnames[4]: nb_of_tiles,
                                     fieldnames[5]: round(top1accuracy, 3),
                                     fieldnames[6]: round(top5accuracy, 3),
                                     fieldnames[7]: str(datetime.timedelta(seconds=time)),
                                     fieldnames[8]: str(datetime.timedelta(seconds=time_get_patches2)),
                                     fieldnames[9]: str(datetime.timedelta(seconds=time_get_features2)),
                                     fieldnames[10]: str(datetime.timedelta(seconds=time_comparison)),
                                     })

                writer.writerow(dict.fromkeys(fieldnames, ""))  # empty line

                writer.writerow({fieldnames[0]: "Total",
                                 fieldnames[1]: "",
                                 fieldnames[2]: "",
                                 fieldnames[3]: total_nb_of_landmarks,
                                 fieldnames[4]: total_nb_of_tiles,
                                 fieldnames[5]: round(total_counter_top1/total_nb_of_landmarks, 3),
                                 fieldnames[6]: round(total_counter_top5/total_nb_of_landmarks, 3),
                                 fieldnames[7]: str(datetime.timedelta(seconds=total_time)),
                                 fieldnames[8]: str(datetime.timedelta(seconds=total_time_get_patches2)),
                                 fieldnames[9]: str(datetime.timedelta(seconds=total_time_get_features2)),
                                 fieldnames[10]: str(datetime.timedelta(seconds=total_time_comparison)),
                                 })

                last_row_dic = dict.fromkeys(fieldnames, "")  # empty line
                last_row_dic.update({fieldnames[0]: 'Mean',
                                     fieldnames[7]: str(datetime.timedelta(seconds=total_time/len(list_top1accuracy))),
                                     fieldnames[8]: str(datetime.timedelta(seconds=total_time_get_patches2/len(list_top1accuracy))),
                                     fieldnames[9]: str(datetime.timedelta(seconds=total_time_get_features2/len(list_top1accuracy))),
                                     fieldnames[10]: str(datetime.timedelta(seconds=total_time_comparison/len(list_top1accuracy))),
                                     })
                writer.writerow(last_row_dic)


if __name__ == "__main__":
    main()
