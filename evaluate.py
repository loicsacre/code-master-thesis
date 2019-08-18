"""Evaluate the performance of trained models on the testing set"""

import csv
import gc
import os
import pickle
from argparse import ArgumentParser
from time import time

import numpy as np
import torch
from PIL import Image
from tabulate import tabulate
from torch.autograd import Variable
from torchvision import transforms

from matchnet.models import (MatchNetEval, TransferAlexNetEval,
                             TransferVGGNetEval, transform_matchnet,
                             transform_transfernet)
from siamese.models import SiameseAlexNetEval
from training_tools import AdaptiveTransformation
from utils import (euclidean_distance, get_patches_from_landmarks,
                   get_position_landmarks, mkdir, segment_image)


def get_pairs():

    with open("./datasets/cnn/dataset-networks.data", 'rb') as input_file:
        dataset = pickle.load(input_file)

    dataset_eval_sim = dataset["testing"]["similar"]

    samples_eval_set = set([(x[0], x[1][0], x[1][1])
                            for x in dataset_eval_sim])

    pairs = []

    pairs_filename = "./results/imagenet/2/pairs.data"
    with open(pairs_filename, 'rb') as output_file:
        pairs_data = pickle.load(output_file)

    for (tissue, dye1, dye2, images_path, original_name1, original_name2, extension) in pairs_data:

        if (tissue, dye1, dye2) in samples_eval_set or (tissue, dye2, dye1) in samples_eval_set:

            pairs.append((tissue, dye1, dye2, images_path,
                          original_name1, original_name2, extension))
            pairs.append((tissue, dye2, dye1, images_path,
                          original_name2, original_name1, extension))

    return pairs


def get_model(arch):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"## Using device {device.type}", flush=True)

    if arch == "matchnet":
        model = MatchNetEval()
    elif arch == "transferAlexnet":
        model = TransferAlexNetEval()
    elif arch == "transferVggnet":
        model = TransferVGGNetEval()
    elif arch == "siameseAlexnet":
        model = SiameseAlexNetEval()

    model.to(device)
    return model


def compare_with_model(model, patches_img1, patches_img2, tissue, dye1, dye2):

    arch = get_args().arch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"## Device used: {device}", flush=True)
    print(f"Evaluating", not model.training)

    start_time = time()

    transform = None
    if arch == "matchnet":
        transform = transform_matchnet
    else:
        transform = AdaptiveTransformation().transform
    comparison = np.zeros((len(patches_img1), len(patches_img2)))

    with torch.no_grad():
        for i, (_, _, ref_patch) in enumerate(patches_img1):

            start_time_single = time()

            ref_patch = Image.fromarray(ref_patch)
            if arch == "matchnet":
                ref_patch = transform(ref_patch)
            else:
                params = {"tissue": tissue, "dye": dye1}
                ref_patch = transform(**params)(ref_patch)

            ref_patch = torch.unsqueeze(ref_patch, 0)
            ref_patch = Variable(ref_patch).to(device)
            model.set_reference(ref_patch)

            for j, (_, _, patch) in enumerate(patches_img2):
                patch = Image.fromarray(patch)
                if arch == "matchnet":
                    patch = transform(patch)
                else:
                    params = {"tissue": tissue, "dye": dye2}
                    patch = transform(**params)(patch)

                patch = torch.unsqueeze(patch, 0)
                patch = Variable(patch).to(device)
                comparison[i, j] = model.forward_with_reference(patch, j).data

            if i < 5:
                print(
                    f"## {i} : time to compare {(time() - start_time_single):3f})", flush=True)

    print(
        f"## Time to compare {(time() - start_time):3f} ({len(patches_img1)}, {len(patches_img2)})", flush=True)
    model.reset()
    return comparison


def get_args():
    """get the arguments of the program"""
    parser = ArgumentParser(
        prog="Evaluate the performance of trained models on the testing set")

    parser.add_argument("--arch", dest='arch',
                        default='matchnet',
                        choices=["matchnet", "transferAlexnet",
                                 "transferVggnet", "siameseAlexnet"],
                        help="model architecture: \
                              matchnet (MatchNet) | \
                              transferAlexnet (TransferAlexNet) | \
                              transferVggnet (TransferVGGNet) | \
                              siameseAlexnet (SiameseAlexNet) | \
                             (default: matchnet)")

    parser.add_argument('--output', dest='output',
                        default="./results/evaluation/",
                        help="Where all the results will be saved")

    parser.add_argument("--size", dest='size',
                        default=300,
                        type=int,
                        help='Size of the patches (in pixels)')

    parser.add_argument("--shift", dest='shift',
                        default=75,
                        type=int,
                        help='Shift for the sliding window \
                            in the segmentation (in pixels)')

    parser.add_argument('--checkpoint', dest='checkpoint',
                        required=True,
                        help="Path to the checkpoint")

    parser.add_argument('-d', dest='d',
                        action="store_true",
                        help="Activate debug mode")

    return parser.parse_args()


def main():
    """main function"""

    args = get_args()
    arch = args.arch

    print("\n#################", flush=True)
    print("### Arguments ###", flush=True)
    print("#################", flush=True)
    for arg in vars(args):
        print(f"{arg} : {getattr(args, arg)}", flush=True)
    print("#################\n", flush=True)

    # Set evaluation mode
    model = get_model(args.arch)
    model.eval()

    output = args.output
    mkdir(output)

    if os.path.isfile(args.checkpoint):
        print(f"## Loading checkpoint..", flush=True)
        if torch.cuda.is_available():
            checkpoint = torch.load(args.checkpoint)
        else:
            checkpoint = torch.load(
                args.checkpoint, map_location='cpu')
        print(f"## Loaded checkpoint '{args.checkpoint}'\n", flush=True)
    else:
        print(f"## No checkpoint found for '{args.checkpoint}'\n", flush=True)
        return

    # Load state of the model
    model.load_state_dict(checkpoint['state_dict'])

    pairs = get_pairs()

    total_counter = [0]*2
    total_nb_of_landmarks = 0

    output = os.path.join(args.output, "cnn", f"{arch}")

    mkdir(output)

    csv_row = []

    print(f"## Number of pairs: {len(pairs)} to compare..", flush=True)

    for i, (tissue, dye1, dye2, images_path, original_name1, original_name2, extension) in enumerate(pairs):

        output_comparison = os.path.join(
            output, f"{tissue}-{dye1}-{dye2}.comp")

        # filename1 : reference (comparison from its annotation to those from filename1) -> get the patches
        # filename2 : target (image to cut)

        print(f"## {i}", tissue, dye1, dye2)

        if not os.path.exists(output_comparison):
            print("## File does not exist", flush=True)

            start_time = time()

            # Get patches from image 1
            patches_img1_landmarks = get_patches_from_landmarks(
                tissue, original_name1, size=args.size)

            # Get patches from image 2
            patches_img2 = segment_image(filename=os.path.join(
                images_path, original_name2 + extension), size=args.size, shift=args.shift)
            # patches_img2 = get_patches_from_landmarks(
            #     tissue, original_name2, size=args.size)

            # get the features
            # number of available landmarks for the particular tissue
            nb_of_landmarks = len(patches_img1_landmarks)
            print("==> Img1 ({} {}) : {}".format(
                tissue, dye1, nb_of_landmarks), flush=True)
            print("==> Img2 ({} {}) : {}".format(
                tissue, dye2, len(patches_img2)), flush=True)

            # Comparison
            start_time_comparison = time()
            results_comparison = compare_with_model(model, patches_img1_landmarks, patches_img2, tissue,
                                                    dye1, dye2)
            model.reset()
            time_comparison = time() - start_time_comparison
            print(f"Comparison time: {time_comparison}")
            del patches_img1_landmarks

            center_patches_img2 = [x[1] for x in patches_img2]
            del patches_img2
            gc.collect()

            state = {
                "time_comparison": time_comparison,
                "results_comparison": results_comparison
            }

            with open(output_comparison, "wb") as output_file:
                pickle.dump(state, output_file)
        else:
            print("## File exists", flush=True)
            nb_of_landmarks = len(get_patches_from_landmarks(
                tissue, original_name1, size=args.size))

            with open(output_comparison, "rb") as output_file:
                state = pickle.load(output_file)

            results_comparison = state["results_comparison"]

            center_patches_img2 = [x[1] for x in segment_image(filename=os.path.join(
                images_path, original_name2 + extension), size=args.size, shift=args.shift)]
            gc.collect()

        # Get top-k accuracy

        k_list = [1, 5]
        # count the landmarks respecting the condition
        counter = [0]*len(k_list)

        position_landmarks_dye2 = get_position_landmarks(
            tissue, original_name2)

        for j in range(nb_of_landmarks):

            array = [(k, x) for k, x in enumerate(results_comparison[j])]
            array.sort(key=lambda x: x[1], reverse=True)

            for c, k in enumerate(k_list):

                if "siamese" in arch:
                    indices_of_best_matches = [x[0] for x in array[-k:]]
                else:
                    indices_of_best_matches = [x[0] for x in array[:k]]

                # get the position of the k centers that best matches
                centers = [center_patches_img2[ind]
                           for ind in indices_of_best_matches]
                true_position = position_landmarks_dye2[j]

                distances = [euclidean_distance(
                    np.array(center), np.array(true_position)) for center in centers]
                distances = np.array(distances)

                # if at least a patch center is within a certain radius around the true landmark
                if distances[distances <= args.size/2].shape[0] != 0:
                    counter[c] += 1
                    total_counter[c] += 1

        table = []
        top_accuracy_list = []

        total_nb_of_landmarks += nb_of_landmarks

        print(f"## Nb of lanfmarks {nb_of_landmarks}")

        for c, k in enumerate(k_list):
            acc = round(counter[c]/nb_of_landmarks, 4)
            top_accuracy_list.append((k, acc))
            table.append([str(k), str(acc)])
        t = tabulate(table, headers=['k', 'Top-k accuracy'])
        print("\n", t, "\n", flush=True)

        csv_row.append([tissue, dye1, dye2, round(
            counter[0]/nb_of_landmarks, 4), round(counter[1]/nb_of_landmarks, 4)])

    print(f"Total top-1: {total_counter[0]/total_nb_of_landmarks}", flush=True)
    print(f"Total top-5: {total_counter[1]/total_nb_of_landmarks}", flush=True)

    csv_row.append(["Total", "", "", round(total_counter[0]/total_nb_of_landmarks,
                                           4), round(total_counter[1]/total_nb_of_landmarks, 4)])

    output_csv_dir = os.path.join(args.output, "final-landmarks")
    mkdir(output_csv_dir)

    csv_prefix_filename = args.checkpoint.split("/")[-1].rsplit(".", 1)[0]
    with open(os.path.join(output_csv_dir, f"{csv_prefix_filename}.csv"), mode='w') as csv_file:
        fieldnames = ['tissue', 'dye1', 'dye2',
                      'top-1-accuracy', 'top-5-accuracy']

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in csv_row:
            row_dict = dict()
            for fieldname, row_el in zip(fieldnames, row):
                row_dict[fieldname] = row_el

            writer.writerow(row_dict)


if __name__ == "__main__":
    main()
