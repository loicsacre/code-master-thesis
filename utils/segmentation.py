import warnings
import os
import csv

import numpy as np
import PIL
from PIL import Image, ImageDraw
from math import cos, sin, radians

from path import Paths

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
PIL.Image.MAX_IMAGE_PIXELS = 933120000

"""
coordinates : (x, y) (PIL convention) as the coordinates of the landmarks

(0,0)    ----  (w-1, 0) 
  |                |
  |                |
  |                |
(0, h-1) ---- (w-1, h-1)

PIL Image: x,y

PIL -> Numpy: y,x,c
PIL -> Tensor: c,y,x
Tensor -> Numpy: c,y,x (np.moveaxis(x, 0, 2) --> y,x,c)
(where c is the channel index)

PIL -> Tensor -> Numpy --> PIL: 
    x = np.moveaxis(x, 0, 2)
    Image.fromarray(x.astype(np.uint8))
"""


def get_center_from_window(xy):
    """
    xy: ((x, y), (x + width, y + height))
    return tuple (xc, yc), the center of xy 
    """
    ((x_min, y_min), (x_max, y_max)) = xy
    xc = (x_min + (x_max-x_min)//2)
    yc = (y_min + (y_max-y_min)//2)
    return (xc, yc)


def get_window_from_center(center, size=300):
    "if x_min or y_min are negative -> replace with 0"

    xc, yc = center
    assert(xc >= 0 and yc >= 0)

    x_min = max(0, xc - size//2)
    x_max = xc + size//2 + 1*size % 2
    y_min = max(0, yc - size//2)
    y_max = yc + size//2 + 1*size % 2

    return ((x_min, y_min), (x_max, y_max))


def get_patch_from_center(img_arr, center, size=300):
    "center : tuple (x, y)"

    ((x_min, y_min), (x_max, y_max)) = get_window_from_center(center, size=size)
    patch = img_arr[y_min: y_max, x_min: x_max, :]

    return patch


def get_coordinates_after_rotation(position, angle, original_size, new_size=None, expand=False):
    """
    position : tuple (x, y)
    angle in degrees
    original_size : (w, h)
    new_size : (w, h) TODO: compute automatically
    expand : True or False
    """
    assert(expand and (new_size is not None))

    theta = radians(angle)

    # Rotation matrix
    R = np.array([[cos(theta), sin(theta)],
                  [-sin(theta), cos(theta)]])

    position = np.array(position)
    for i in range(2):
        position[i] -= original_size[i]/2
    new_position = R.dot(position.T)
    
    for i in range(2):
        new_position[i] += original_size[i]/2
        if expand:
            new_position[i] += (new_size[i] - original_size[i])/2

    return tuple(new_position)


def get_patch(img_arr, xy):
    "xy: ((x, y), (x + width, y + height))"

    ((x_min, y_min), (x_max, y_max)) = xy
    patch = img_arr[y_min: y_max, x_min: x_max, :]

    return patch


def segment_image(filename=None, img=None, size=300, shift=None, transpose=None):
    assert((filename is None) != (img is None))

    if shift is None:
        shift = size//4

    if filename is not None:
        print("### Segmenting file:", filename)
        img = Image.open(filename)

    if transpose is not None:
        print(f"## Applying transpose : {transpose}")
        img = img.transpose(transpose)
    width, height = img.size
    img_arr = np.array(img)
    print("## size (w, h) ", width, height)

    patches = []

    for x_min in range(0, width, shift):
        for y_min in range(0, height, shift):

            xy = ((x_min, y_min), (x_min + size, y_min + size))
            patch = get_patch(img_arr, xy)

            w_patch, h_patch, c_patch = patch.shape

            if w_patch != size or h_patch != size:
                patch_tmp = np.zeros((size, size, c_patch),
                                     dtype=np.uint8)

                for c in range(c_patch):
                    patch_tmp[:, :, c] = patch_tmp[:, :, c] + \
                        np.mean(patch[:, :, c])

                patch_tmp[0:w_patch, 0:h_patch, :] = patch
                patch = patch_tmp

            # Condition to accept the patch
            if not(patch.mean() > 210 and patch.std() < 6):

                center = (x_min + size//2, y_min + size//2)
                ((x_min, y_min), (x_max, y_max)) = xy
                patches.append((xy, center, patch))

            # TODO: remove
            # if len(patches) > 7:
            #     return patches

    if filename is not None:
        img.close()
    print("--> number of accepted patches:", len(patches))

    return patches


def visualize_segmentation(filename, output='result', size=300, shift=None, transpose=None):

    image = Image.open(filename)
    if transpose is not None:
        image = image.transpose(transpose)

    patches = segment_image(filename=filename, size=size,
                            shift=shift, transpose=transpose)

    image = image.convert("RGBA")
    tmp = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # Create a drawing context for it.
    draw = ImageDraw.Draw(tmp)

    for xy, _, _ in patches:
        args = {"xy": xy, "outline": "black", "fill": (
            0, 0, 255, 127) if size == 300 else (255, 0, 0, 127)}
        draw.rectangle(**args)

    # Alpha composite the two images together.
    img = Image.alpha_composite(image, tmp)
    tmp.close()
    img = img.convert("RGB")
    img.save(f"{output}.jpg")


def get_position_landmarks(tissue, original_name):

    # Get the path to the annotations CSV
    annotations_path = Paths.PATH_TO_LANDMARKS + tissue + "/"

    entries = os.listdir(annotations_path)
    # Remove .DS_Store file (esp. for MacOS)
    if '.DS_Store' in entries:
        entries.remove('.DS_Store')

    annotations_path += entries[0] + \
        "/" + original_name + ".csv"

    # list of (x,y) coordinates corresponding the position of the landmarks
    positions = []

    with open(annotations_path, 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        next(f_csv)

        for _, x, y in f_csv:

            center = (int(float(x)), int(float(y)))
            positions.append(center)

    return positions


def get_patches_from_landmarks(tissue, original_name, size=300):
    """ Get the patches in a local way """

    # Get the path to the annotations CSV
    annotations_path = Paths.PATH_TO_LANDMARKS + tissue + "/"

    entries = os.listdir(annotations_path)
    # Remove .DS_Store file (esp. for MacOS)
    if '.DS_Store' in entries:
        entries.remove('.DS_Store')

    annotations_path += entries[0] + \
        "/" + original_name + ".csv"

    # Get the path to the images
    image_path = Paths.PATH_TO_IMAGES + tissue + \
        "/" + entries[0] + "/" + original_name

    extension = ".jpg"
    if not os.path.exists(image_path + extension):
        extension = ".png"
    image_path += extension

    patches = []

    position_landmarks = get_position_landmarks(tissue, original_name)

    with Image.open(image_path) as image:

        img_arr = np.array(image)

        for center in position_landmarks:

            patch = get_patch_from_center(img_arr, center, size=size)
            xy = get_window_from_center(center, size=size)
            patches.append((xy, center, patch))

    return patches


class Divider():

    def __init__(self, filename, size, shift):
        self.filename = filename
        self.size = size
        self.shift = shift

        self.img = img = Image.open(filename)
        width, height = img.size
        self.img_arr = np.array(img)
        self.img.close()
        self.x_min = 0
        self.y_min = 0
        # range(0, width, shift)[-1]
        self.x_max = (width//self.shift)*self.shift
        # range(0, height, shift)[-1]
        self.y_max = (height//self.shift)*self.shift
        self.number_of_patches = 0
        self.stop = False

        self.counter = 0

    def get_patch(self):

        patch = None
        patch_found = False

        while True:

            xy = ((self.x_min, self.y_min),
                  (self.x_min + self.size, self.y_min + self.size))
            patch = get_patch(self.img_arr, xy)

            w_patch, h_patch, c_patch = patch.shape

            if w_patch != self.size or h_patch != self.size:
                patch_tmp = np.zeros((self.size, self.size, c_patch),
                                     dtype=np.uint8)

                for c in range(c_patch):
                    patch_tmp[:, :, c] = patch_tmp[:, :, c] + \
                        np.mean(patch[:, :, c])

                patch_tmp[0:w_patch, 0:h_patch, :] = patch
                patch = patch_tmp

            # Condition to accept the patch
            if not(patch.mean() > 210 and patch.std() < 6):

                center = (self.x_min + self.size//2, self.y_min + self.size//2)
                patch = (xy, center, patch)
                self.number_of_patches += 1
                patch_found = True

            self.counter += 1
            self.y_min += self.shift
            if self.y_max < self.y_min:
                self.x_min += self.shift
                if self.x_max < self.x_min:
                    self.stop = True
                    # print("COUNTER", self.counter, patch_found)
                    return None

                self.y_min = 0

            if patch_found:
                break

        # print("Yeah", self.number_of_patches, self.x_min, self.x_max, self.y_min, self.y_max, self.stop)
        return patch
