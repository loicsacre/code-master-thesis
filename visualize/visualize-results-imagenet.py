"""
Visualize a tissue section with the results from the comparisons by applying 
a layer with the patches from the segmentation coloured with a color 
corresponding to the similarity score

Note:
    - it is only working with the comparison results from the networks pre-trained on ImageNet (i.e. imagenet, experience 2).
Nevertheless, it could be very easily adapated to perform the same task with personal models. 
    - Make sure that the .data file is generated in ./results/imagenet/2/data/ (or elsewhere)
"""

import csv
import errno
import os
import pickle
import warnings
from argparse import ArgumentParser

import numpy as np
import torchvision.models as models
from PIL import Image, ImageDraw

from path import Paths
from utils import (draw_rectangle_with_value, get_position_landmarks,
                   get_window_from_center, mkdir, segment_image,
                   get_coordinates_after_rotation)

warnings.simplefilter('ignore', Image.DecompressionBombWarning)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_args():
    """get the parameters of the program"""

    parser = ArgumentParser(prog="Main")

    parser.add_argument('--path_to_patches', dest='path_to_patches',
                        default=Paths.PATH_TO_PATCHES,
                        help="The path to the crops")

    parser.add_argument('--info', dest='info',
                        default='./info/project-info.csv',
                        help="The path the project info csv")

    parser.add_argument('--data', dest='data',
                        default='./results/imagenet/2/data/',
                        help="The path to the data")

    parser.add_argument('--size', dest='size',
                        type=int,
                        default=300,
                        help="The size of the patches")

    parser.add_argument("--arch", dest='arch',
                        default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

    parser.add_argument("--distance", dest='distance',
                        default='cos',
                        choices=["cos", "eucl", "eucl-norm"],
                        help='Distance metric : \
                        cosine similarity (cos) | \
                        euclidean distance (eucl) \
                        euclidean distance with normalized vectors (eucl-norm) \
                        (default: cos)')

    parser.add_argument('--landmarks', dest='landmarks',
                        default=Paths.PATH_TO_LANDMARKS,
                        help="The path where the annotations are stored")

    parser.add_argument('--path_to_images', dest='path_to_images',
                        default=Paths.PATH_TO_IMAGES,
                        help="The path where the images are stored")

    parser.add_argument('--output', dest='output',
                        default="./results/imagenet/2/visualize",
                        help="The path where the result image will be stored")

    parser.add_argument('--tissue', dest='tissue',
                        type=str,
                        required=True,
                        help="The tissue name")

    parser.add_argument('--dye1', dest='dye1',
                        type=str,
                        required=True,
                        help="The dye of the reference")

    parser.add_argument('--dye2', dest='dye2',
                        type=str,
                        required=True,
                        help="The dye of the reference")

    parser.add_argument('-pool', dest='pool',
                        action="store_true",
                        help="Indicates whether the pooling was applied")

    parser.add_argument('--references', dest='references',
                        nargs='+', type=int,
                        help='The number of the referent annotations',
                        default=[0])

    parser.add_argument('--angle', dest='angle',
                        type=int,
                        help='Indicates if the target image has to be rotated of <angle> degrees',
                        default=None)

    return parser.parse_args()


def main():

    args = get_args()

    output_dir = "visualize/imagenet/{args.tissue}/"
    mkdir(output_dir)

    path_to_data = os.path.join(
        args.data, args.distance, str(args.size), args.arch)

    counter = 0  # check that the dyes exist

    original_name = None
    scale = None

    with open(args.info, 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        next(f_csv)

        for row in f_csv:

            tissue = row[1]
            dye = row[2]

            if args.tissue == tissue:
                if args.dye1 == dye:
                    counter += 1
                if args.dye2 == dye:
                    counter += 1
                    original_name = row[6]
                    scale = row[7]

    if counter != 2:
        print(f"Check the entered dyes {args.dye1} and {args.dye2}...")
        return

    # Get the path to the dir where are the images
    path_to_images_dir = os.path.join(args.path_to_images, args.tissue, scale)

    # Try to find the image the image format (png or jpg)
    extension = ".jpg"
    if not os.path.exists(os.path.join(path_to_images_dir, original_name + extension)):
        extension = ".png"
    path_to_target_image = os.path.join(path_to_images_dir, original_name + extension)

    path_to_data = os.path.join(
        path_to_data, f"{args.tissue}_{args.dye1}_{args.dye2}_{args.arch}_{args.size}_{args.pool}.data")

    with open(path_to_data, "rb") as file_data:
        data = pickle.load(file_data)
        shift = data["args"]["shift"]

    results_comparison = data["results_comparison"]

    img_o = None
    if args.angle is not None:
        img_o = Image.open(path_to_target_image)
        im2 = img_o.convert('RGBA')
        # rotated image
        rot = im2.rotate(args.angle, expand=True)
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, (255,)*4)
        # create a composite image using the
        out = Image.composite(rot, fff, rot)
        out = out.convert(img_o.mode)
        patches_list = segment_image(img=out, size=args.size, shift=shift)
        img = out.convert('RGBA')
    else:
        img_o = Image.open(path_to_target_image)
        img = img_o.convert('RGBA')

    # Get the list of all the patches
    patches_list = segment_image(path_to_target_image, size=args.size, shift=shift)

    ind = np.unravel_index(
        np.argmax(results_comparison, axis=None), results_comparison.shape)
    print("Best landmark", results_comparison[ind], ind)
    ind = np.unravel_index(
        np.argmin(results_comparison, axis=None), results_comparison.shape)
    print("Worst landmark", results_comparison[ind], ind)

    for r in args.references:

        max_landmark_nb = np.argmax(
            results_comparison[r])  # the best match
        min_landmark_nb = np.argmin(
            results_comparison[r])

        maxi = results_comparison[r][max_landmark_nb]
        mini = results_comparison[r][min_landmark_nb]
        print(f"Minimum/Maximum similarity value : {mini}/{maxi}")

        # sort the patches (for displaying the best above)
        array = [(k, x) for k, x in enumerate(results_comparison[r])]
        array.sort(key=lambda x: x[1])

        percentiles = np.percentile([x[1] for x in array], [
                                    10, 25, 50, 75, 90, 95])
        # The percentile can be chosen easily
        colrange = [percentiles[4], maxi]

        # Make a blank image for the rectangle, initialized to a completely
        # transparent color.
        tmp = Image.new('RGBA', img.size, (0, 0, 0, 0))

        # Create a drawing context for it.
        draw = ImageDraw.Draw(tmp)

        for i, _ in array:

            value = results_comparison[r][i]
            _, center, _ = patches_list[i]

            params = {"context": draw,
                      "value": round(value, 2),
                      "center": center,
                      "size": args.size,
                      "colrange": colrange}

            if i in [x[0] for x in array[-5:-1]]:
                params["ismax"] = True
            if i in [x[0] for x in array[-1:]]:
                params["isfirst"] = True

            draw_rectangle_with_value(**params)

        # Alpha composite the two images together.
        img_final = Image.alpha_composite(img, tmp)
        tmp.close()

        tmp = Image.new('RGBA', img_final.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(tmp)

        # Draw a rectangle where the patch should be located
        position_landmarks_dye = get_position_landmarks(
            args.tissue, original_name)

        # Rotate the reference position too if needed
        if args.angle is not None:
            position_reference = get_coordinates_after_rotation(
                position_reference, args.angle, img_o.size, img_final.size, True)
        else:
            position_reference = position_landmarks_dye[r]

        xy_reference = get_window_from_center(
            position_reference, size=args.size)
     
        params = {"xy": xy_reference, "outline": "yellow", "width": 10}
        ((x, y), (x_r, y_r)) = xy_reference
        draw.line(xy_reference, fill="yellow", width=10)
        draw.line(((x, y_r), (x_r, y)), fill="yellow", width=10)
        draw.rectangle(**params)

        # Alpha composite the two images together.
        img_final = Image.alpha_composite(img_final, tmp)
        tmp.close()

        # Remove alpha for saving in jpg format.
        img_final = img_final.convert("RGB")

        print(f"## Saving output image (reference {r})..")
        img_final.save(os.path.join(output_dir, f"{args.dye1}-{args.dye2}-{r}.jpg"))

    img.close()


if __name__ == '__main__':
    main()
