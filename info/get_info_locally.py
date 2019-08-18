
import csv
import os
import warnings

from PIL import Image
import PIL
from path import Paths

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
PIL.Image.MAX_IMAGE_PIXELS = 933120000


tissue2info = {
    "lung-lesion": (40, 0.174),
    "lung-lobes": (10, 1.274),
    "mammary-gland": (10, 2.294),
    "mice-kidney": (20, 0.227),
    "COAD": (10, 0.468),
    "gastric": (40, 0.2528),
    "breast": (40, 0.2528),
    "kidney": (40, 0.2528)
}


def main():

    with open('./info/project-info.csv', mode='w') as csv_file:

        fieldnames = ['Image ID', 'Tissue', 'Dye', 'Height', 'Width', 'Number of Landmarks',
                      "Original Name", "Scale", "Magnitude", "Resolution (µm/pixel)"]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # add the dyes for samples which does not have explicit dyes as name, such as 29-041-Izd2-w35-Ki67-7-les2
        dyes = ["HE", 'HER2', 'CC10', "PRO-SPC", "KI67", "CD31", "ER",
                "PR", "PAS", "MAS", "SMA", "NEU", "CD4", "CD68", "EBV", "PROSPC"]

        tissue2dye = dict()  # for handling cases where there is the twice the dye for a tissue

        walk_path = iter(os.walk(Paths.PATH_TO_IMAGES))
        next(walk_path)

        counter = 0
        for root, dirs, files in walk_path:

            if not dirs:

                path_to_landmarks = root.replace(
                    Paths.PATH_TO_IMAGES, Paths.PATH_TO_LANDMARKS)

                if os.path.exists(path_to_landmarks):

                    tissue = root.split("/")[-2]
                    scale = root.split("/")[-1]

                    for f in files:

                        # original_name.extension
                        original_name = f.split(".")[0]

                        path_to_landmarks_img = os.path.join(
                            path_to_landmarks, f"{original_name}.csv")

                        if not os.path.exists(path_to_landmarks_img):
                            continue

                        with open(path_to_landmarks_img, mode='r') as csv_file:
                            f_csv = csv.reader(
                                csv_file, delimiter=str(','), quotechar=str('|'))
                            next(f_csv)
                            nb_of_landmarks = sum([1 for f in f_csv])

                        path_to_img = os.path.join(root, f)
                        img = Image.open(path_to_img)

                        width, height = img.size

                        # get the dye in uppercase
                        possible_dyes = list(
                            filter(lambda k: k in original_name.upper(), dyes))

                        if len(possible_dyes) != 0:
                            # for example : lung-lesion_3 PRO-SPC 29-041-Izd2-w35-proSPC-4-les3 --> possible_dyes = ['PR', 'PROSPC']
                            dye = max(possible_dyes, key=len)

                            # Special case for lung-lesion_1_29-041-Izd2-w35-proSPC-4-les1_pyr.tif
                            if dye is "PROSPC":
                                dye = "PRO-SPC"
                        else:
                            dye = original_name.upper()

                        if tissue not in tissue2dye:
                            tissue2dye[tissue] = set()

                        if dye not in tissue2dye[tissue]:
                            tissue2dye[tissue].add(dye)
                        else:
                            # add the id to differentiate the same dye appearing twice
                            dye += "." + str(counter)

                        print(tissue, dye, original_name)

                        magnitude = tissue2info[tissue.split("_")[0]][0]
                        resolution = tissue2info[tissue.split("_")[0]][1]

                        writer.writerow({fieldnames[0]: counter,
                                         fieldnames[1]: tissue,
                                         fieldnames[2]: dye,
                                         fieldnames[3]: height,
                                         fieldnames[4]: width,
                                         fieldnames[5]: nb_of_landmarks,
                                         fieldnames[6]: original_name,
                                         fieldnames[7]: scale,
                                         fieldnames[8]: magnitude,
                                         fieldnames[9]: resolution,
                                         })
                        counter += 1


if __name__ == '__main__':
    main()
