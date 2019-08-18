"""
Main program for generating ImageNet results for experiment 1
"""

import os
import pickle
from argparse import ArgumentParser

import torchvision.models as models

from accuracy import get_accuracy
from compare import compare
from features import get_features
from path import Paths
from utils import mkdir

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_args():
    """get the parameters of the program"""
    parser = ArgumentParser(prog="Program to extract features")

    parser.add_argument('--path_to_patches', dest='path_to_patches',
                        default='./datasets/patches',
                        help="The path to the patches")

    parser.add_argument('--output', dest='output',
                        default='./results/imagenet/1/',
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
                        help='Size of the patches')

    parser.add_argument("--distance", dest='distance',
                        default='cos',
                        choices=["cos", "eucl", "eucl-norm"],
                        help='Distance metric : \
                        cosine similarity (cos) | \
                        euclidean distance (eucl) \
                        euclidean distance with normalized vectors (eucl-norm) \
                        (default: cos)')

    parser.add_argument('-pool', dest='pool',
                        action="store_true",
                        help="Indicates whether the pooling must be applied")

    return parser.parse_args()


def get_prefix_file():
    args = get_args()
    output = os.path.join(args.output, args.arch, str(args.size))
    mkdir(output)
    return output + "/" + args.arch + "-" + str(args.size)


def main():
    """main function"""

    args = get_args()

    print("\n#################")
    print("### Arguments ###")
    print("#################")
    for arg in vars(args):
        print(f"{arg} : {getattr(args, arg)}")
    print("#################\n")

    # get the features

    output_features_filename = get_prefix_file() + ".features"

    if not os.path.exists(output_features_filename) or True:
        
        results_features = get_features(args.arch, 300, pooling=args.pool)
        with open(output_features_filename, 'wb') as output_file:
            pickle.dump(results_features, output_file)
    else:
        print("## features already generated")
        return

    # generate the similarity measures for patch-pair combinations
    results_comparison = compare(results_features, args.distance)

    output_comparison_filename = get_prefix_file() + ".comparison"
    with open(output_comparison_filename, 'wb') as output_file:
        pickle.dump(results_comparison, output_file)

    # generate the top-k accuracy (for k = 1 and 5)
    top1accuracy, top5accuracy = get_accuracy(
        results_comparison, args.distance)
    print("Top-1 and 5 accuracy for size {} : {} and {}\n\n".format(args.size,
                                                                    top1accuracy, top5accuracy))


if __name__ == "__main__":
    main()
