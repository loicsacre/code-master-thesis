import datetime
import gc
import os
import pickle
from argparse import ArgumentParser
from time import time

import numpy as np
import torchvision.models as models
from PIL import Image
from tabulate import tabulate

from info import generate_pairs
from path import Paths
from utils import (FeaturesExtractor, MemoryFollower, Normalizer, compare,
                   euclidean_distance, get_patches_from_landmarks,
                   get_position_landmarks, mkdir, segment_image)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_args():
    """get the parameters of the program"""
    parser = ArgumentParser(prog="Program to extract features")

    parser.add_argument('--path_to_patches', dest='path_to_patches',
                        default=Paths.PATH_TO_PATCHES,
                        help="The path to the patches")

    parser.add_argument('--output', dest='output',
                        default='./results/imagenet/2',
                        help="Where all the output results will be stored")

    # https://pytorch.org/docs/stable/torchvision/models.html
    parser.add_argument("--arch", dest='arch',
                        default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

    parser.add_argument("--size", dest='size',
                        default=300,
                        type=int,
                        help='Size of the patches (in pixels)')

    parser.add_argument("--shift", dest='shift',
                        default=75,
                        type=int,
                        help='Shift for the sliding window \
                            in the segmentation (in pixels)')

    parser.add_argument("--distance", dest='distance',
                        default='cos',
                        choices=["cos", "eucl", "eucl-norm"],
                        help='Distance metric : \
                        cosine similarity (cos) | \
                        euclidean distance (eucl) \
                        euclidean distance with normalized vectors (eucl-norm) \
                        (default: cos)')

    parser.add_argument('--resize', dest='resize',
                        default=None,
                        type=int,
                        help="Indicates whether the patch must be resized to <resize> X <resize> \
                        before going through the feature extractor")

    parser.add_argument('-pool', dest='pool',
                        action="store_true",
                        help="Indicates whether the pooling must be applied")

    parser.add_argument('-d', dest='d',
                        action="store_true",
                        help="Indicates whether the debug mode is on")

    return parser.parse_args()


def get_features(list_patches_img, normalize):
    """Return a numpy array of the features"""

    args = get_args()
    arch = args.arch
    pool = args.pool
    resize = args.resize
    if resize is None:
        resize = args.size

    features_extractor = FeaturesExtractor(arch=arch)
    features_extractor.set_normalize(normalize)  # Normalize according to dye

    features = np.zeros(len(list_patches_img), dtype=np.float32)

    for i, (_, _, patch) in enumerate(list_patches_img):

        img = Image.fromarray(patch, mode='RGB')
        f = features_extractor.get_features_from_img(
            img, resize, pool).cpu().numpy()

        if i == 0:
            features = np.zeros(
                (len(list_patches_img), f.shape[0]), dtype=np.float32)

        features[i] = f

    return features


def main():
    """main function"""

    args = get_args()

    shift = args.shift

    normalizer = Normalizer()

    print("\n#################")
    print("### Arguments ###")
    print("#################")
    for arg in vars(args):
        print(f"{arg} : {getattr(args, arg)}")
    print("#################\n")

    # creating the pairs of dyes to analyze
    pairs = generate_pairs()

    for i, (tissue, dye1, dye2, images_path, original_name1, original_name2, extension) in enumerate(pairs):

        # Each element of the pairs will play the role of the target
        for s in range(2):

            if s == 1:
                dye1, dye2 = dye2, dye1
                original_name1, original_name2 = original_name2, original_name1

            start_time = time()

            output_filename = os.path.join(
                args.output, f"data/{args.distance}/{args.size}/{args.arch}/{tissue}_{dye1}_{dye2}_{args.arch}_{args.size}_{args.pool}_{args.resize}.data")
            if not os.path.exists(output_filename):
                print(f"File {output_filename} does not exist")
                mkdir(os.path.dirname(output_filename))
            else:
                print(f"File {output_filename} exists\n")
                continue

            # filename1 : reference (comparison from its annotation to those from filename1) -> get the patches
            # filename2 : image to cut

            print(tissue, dye1, dye2, Paths.PATH_TO_IMAGES,
                  original_name1, original_name2)

            # Get patches from image 1
            start_time_get_patches1 = time()
            patches_img1_landmarks = get_patches_from_landmarks(
                tissue, original_name1, size=get_args().size)
            time_get_patches1 = time() - start_time_get_patches1

            # Get patches from image 2
            start_time_get_patches2 = time()
            patches_img2 = segment_image(os.path.join(
                images_path, original_name2 + extension), size=get_args().size, shift=shift)
            time_get_patches2 = time() - start_time_get_patches2

            #################
            # # Is useful to make to have the results for one pair whose target has to be rotated

            # angle = -75
            # img = img = Image.open(images_path + original_name2 + extension)
            # im2 = img.convert('RGBA')
            # # rotated image
            # rot = im2.rotate(angle, expand=1)
            # # a white image same size as rotated image
            # fff = Image.new('RGBA', rot.size, (255,)*4)
            # # create a composite image using the
            # out = Image.composite(rot, fff, rot)
            # out = out.convert(img.mode)
            # patches_img2 = segment_image(img=out, size=get_args().size, shift=shift)
            # time_get_patches2 = time() - start_time_get_patches2
            ##################

            # get the features
            # number of available landmarks for the particular tissue
            nb_of_landmarks = len(patches_img1_landmarks)
            print("==> Img1 ({} {}) : {}".format(
                tissue, dye1, nb_of_landmarks))
            print("==> Img2 ({} {}) : {}".format(
                tissue, dye2, len(patches_img2)))

            start_time_features_img1_landmarks = time()
            normalize_dye1 = normalizer.get(tissue, dye1)
            features_img1_landmarks = get_features(
                patches_img1_landmarks, normalize_dye1)
            time_get_features1 = time() - start_time_features_img1_landmarks
            patches_img1_landmarks = ""
            del patches_img1_landmarks
            gc.collect()

            start_time_features_img2_landmarks = time()
            normalize_dye2 = normalizer.get(tissue, dye2)
            features_img2 = get_features(patches_img2, normalize_dye2)
            time_get_features2 = time() - start_time_features_img2_landmarks
            feature_size = features_img1_landmarks.shape[1]
            print("===> Features size : {}".format(feature_size))

            # Keep only the center and coordinates of patches_img2

            patches_img2 = [(x[0], x[1]) for x in patches_img2]
            gc.collect()

            # Compare

            start_time_comparison = time()
            results_comparison = compare(
                features_img1_landmarks, features_img2, args.distance)
            time_comparison = time() - start_time_comparison
            features_img2 = ""
            del features_img2
            features_img1_landmarks = ""
            del features_img1_landmarks
            gc.collect()

            # Get the position of the landmarks of dye2

            start_time_position_landmarks = time()
            position_landmarks_dye2 = get_position_landmarks(
                tissue, original_name2)
            time_position_landmarks = time() - start_time_position_landmarks

            # Get top-k accuracy

            start_time_get_accuracy = time()

            k_list = [1, 5]
            # count the landmarks respecting the condition
            counter = [0]*len(k_list)

            for i in range(nb_of_landmarks):

                array = [(k, x) for k, x in enumerate(results_comparison[i])]
                array.sort(key=lambda x: x[1], reverse=True)

                for c, k in enumerate(k_list):

                    indices_of_best_matches = None
                    if args.distance == "cos":
                        indices_of_best_matches = [x[0] for x in array[:k]]
                    elif args.distance == "eucl" or args.distance == "eucl-norm":
                        indices_of_best_matches = [x[0] for x in array[-k:]]

                    # get the position of the k centers that best matches
                    centers = [patches_img2[ind][1]
                               for ind in indices_of_best_matches]
                    true_position = position_landmarks_dye2[i]

                    distances = [euclidean_distance(
                        np.array(center), np.array(true_position)) for center in centers]
                    distances = np.array(distances)

                    # if at least a patch center is within a certain radius around the true landmark
                    if distances[distances <= args.size/2].shape[0] != 0:
                        counter[c] += 1

            table = []
            top_accuracy_list = []
            for c, k in enumerate(k_list):
                acc = round(counter[c]/nb_of_landmarks, 4)
                top_accuracy_list.append((k, acc))
                table.append([str(k), str(acc)])
            t = tabulate(table, headers=['k', 'Top-k accuracy'])
            print("\n", t, "\n")

            time_get_accuracy = time() - start_time_get_accuracy
            elapsed_time = time() - start_time

            table = [['Patches image 1', str(datetime.timedelta(seconds=time_get_patches1))],
                     ['Patches image 2', str(
                         datetime.timedelta(seconds=time_get_patches2))],
                     ['Features image 1', str(
                         datetime.timedelta(seconds=time_get_features1))],
                     ['Features image 2', str(
                         datetime.timedelta(seconds=time_get_features2))],
                     ['Position landmarks image 2', str(
                         datetime.timedelta(seconds=time_position_landmarks))],
                     ['Comparison', str(datetime.timedelta(
                         seconds=time_comparison))],
                     ['Compute accuracy', str(
                         datetime.timedelta(seconds=time_get_accuracy))],
                     ['Elapsed time', str(datetime.timedelta(seconds=elapsed_time))]]
            t = tabulate(table, headers=['', 'Time (h:m:s)'])
            print(t, "\n")

            info = {
                "args": vars(args),
                "pair": (tissue, dye1, dye2, images_path, original_name1, original_name2, extension),
                "results_comparison": results_comparison,
                "nb_of_landmarks": nb_of_landmarks,
                "feature_size": feature_size,
                "counter": counter,
                "top_accuracy_list": top_accuracy_list,
                "time": elapsed_time,
                "time_get_patches1": time_get_patches1,
                "time_get_patches2": time_get_patches2,
                "time_get_features1": time_get_features1,
                "time_get_features2": time_get_features2,
                "time_position_landmarks": time_position_landmarks,
                "time_comparison": time_comparison,
                "time_get_accuracy": time_get_accuracy,
            }

            with open(output_filename, 'wb') as output_file:
                pickle.dump(info, output_file)


if __name__ == "__main__":
    main()
