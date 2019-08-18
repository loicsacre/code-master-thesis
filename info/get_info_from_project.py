# -*- coding: utf-8 -*-

"""
Get the all the info from a project

If the images were uploaded to the project with the following format 'tissue&scale&original_name'
Note that the images from  https://anhir.grand-challenge.org/Download/ are stored like : tissue/scale/original_name.ext (where ext is either jpg or png)
"""

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
from argparse import ArgumentParser

import csv
from path import Paths

from cytomine import Cytomine
from cytomine.models.image import ImageInstanceCollection


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

if __name__ == '__main__':
    parser = ArgumentParser(
        prog="Get the project info and save it in a csv file")

    # Cytomine
    parser.add_argument('--cytomine_host', dest='host',
                        default='research.cytomine.be', help="The Cytomine host")
    parser.add_argument('--cytomine_public_key', dest='public_key',
                        help="The Cytomine public key")
    parser.add_argument('--cytomine_private_key', dest='private_key',
                        help="The Cytomine private key")
    parser.add_argument('--cytomine_id_project', dest='id_project',
                        help="The project from which we want the images")

    parser.add_argument('--path_to_images', dest='path_to_images',
                        default=Paths.PATH_TO_IMAGES,
                        help="The path to the images")

    params, _ = parser.parse_known_args(sys.argv[1:])

    with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key,
                  verbose=logging.INFO) as cytomine:

        image_instances = ImageInstanceCollection(
        ).fetch_with_filter("project", params.id_project)

        with open('./info/project-info.csv', mode='w') as csv_file:

            fieldnames = ['Image ID', 'Tissue', 'Dye', 'Height', 'Width', 'Number of Landmarks',
                          "Original Name", "Scale", "Magnitude", "Resolution (µm/pixel)"]

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            # add the dyes for samples which does not have explicit dyes as name, such as 29-041-Izd2-w35-Ki67-7-les2
            dyes = ["HE", 'HER2', 'CC10', "PRO-SPC", "KI67", "CD31", "ER",
                    "PR", "PAS", "MAS", "SMA", "NEU", "CD4", "CD68", "EBV", "PROSPC"]

            tissue2dye = dict()  # for handling cases where there is the twice the dye for a tissue

            for image in image_instances:

                """
                Example to understand the parsing (when the images were uploaded, tissue and original name joined with '_')

                image.filename : /1566135909799/lung-lobes_3&scale-100pc&29-040-U-35W-Izd1-4-cc10_pyr.tif

                original_name : 29-041-Izd2-w35-Ki67-7-les2
                tissue : lung-lesion_2
                dye : KI67
                scale : scale-25pc

                """

                # image.filename : /1566135909799/lung-lobes_3&scale-100pc&29-040-U-35W-Izd1-4-cc10_pyr.tif
                # lung-lobes_3&scale-100pc&29-040-U-35W-Izd1-4-cc10_pyr.tif (remove the path)
                tmp = image.filename.split('/')[2]
                # lung-lobes_3&scale-100pc&29-040-U-35W-Izd1-4-cc10_pyr (remove the '.tif')
                tmp = tmp.split('.')[0]
                # lung-lobes_3&scale-100pc&29-040-U-35W-Izd1-4-cc10 (remove the '_pyr')
                tmp = tmp.rsplit('_', 1)[0]

                tmp = tmp.split("&")
                tissue = tmp[0]
                scale = tmp[1]
                original_name = tmp[2]

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
                    dye += "." + str(image.id)

                print(tissue, dye, original_name)

                magnitude = tissue2info[tissue.split("_")[0]][0]
                resolution = tissue2info[tissue.split("_")[0]][1]

                writer.writerow({fieldnames[0]: image.id,
                                 fieldnames[1]: tissue,
                                 fieldnames[2]: dye,
                                 fieldnames[3]: image.height,
                                 fieldnames[4]: image.width,
                                 fieldnames[5]: image.numberOfAnnotations,
                                 fieldnames[6]: original_name,
                                 fieldnames[7]: scale,
                                 fieldnames[8]: magnitude,
                                 fieldnames[9]: resolution,
                                 })
