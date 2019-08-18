"""
This code assumes that a csv file with the results is already generated thanks to imagenet/2/summary-distance.py
"""


import csv
from argparse import ArgumentParser
import pickle
import os


def get_args():
    """get the arguments of the program"""
    parser = ArgumentParser()

    parser.add_argument("--csv", dest='csv',
                        default="results/imagenet/2/csv/cos/300/densenet201.csv",
                        help="The path to the csv files")

    return parser.parse_args()


def main():
    args = get_args()

    with open("./datasets/cnn/dataset-networks.data", 'rb') as input_file:
        dataset = pickle.load(input_file)

    dataset_test_sim = dataset["testing"]["similar"]
    samples_test_set = set([(x[0], x[1][0], x[1][1])
                            for x in dataset_test_sim])

    total_nb_of_landmark = 0
    counter = [0]*2

    with open(args.csv, 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        next(f_csv)

        for row in f_csv:

            tissue = row[0]

            # end of results
            if tissue == "":
                break

            dye1 = row[1]
            dye2 = row[2]
            nb_of_landmark = int(row[3])
            top1accuracy = float(row[5])
            top5accuracy = float(row[6])

            if (tissue, dye1, dye2) in samples_test_set or (tissue, dye2, dye1) in samples_test_set:
                total_nb_of_landmark += nb_of_landmark
                counter[0] += (top1accuracy*nb_of_landmark)
                counter[1] += (top5accuracy*nb_of_landmark)

    print(args.csv)
    print(f"Top-1-accuracy {counter[0]/total_nb_of_landmark}")
    print(f"Top-5-accuracy {counter[1]/total_nb_of_landmark}")

    print(total_nb_of_landmark)

if __name__ == "__main__":
    main()
