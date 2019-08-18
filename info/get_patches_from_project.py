
# -*- coding: utf-8 -*-

"""
Script for getting the cropped part around around a specific landmark on a image

Tissue
    |----- Dye
            |----- Landmark Number(.format)

E.g.

COAD_1
    |----- S1
            |----- 0.jpg
            |----- 1.jpg
            |----- ...
    |----- S6
            |----- 0.jpg
            |----- 1.jpg
            |----- ...
    |----- ...
    |----- ...
...
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

import os
import csv

from cytomine import Cytomine
from cytomine.models import AnnotationCollection, Annotation, PropertyCollection, Property

if __name__ == '__main__':
    parser = ArgumentParser(prog="Cytomine Python client example")

    # Cytomine
    parser.add_argument('--cytomine_host', dest='host',
                        default='research.cytomine.be', help="The Cytomine host")
    parser.add_argument('--cytomine_public_key', dest='public_key',
                        help="The Cytomine public key")
    parser.add_argument('--cytomine_private_key', dest='private_key',
                        help="The Cytomine private key")
    parser.add_argument('--cytomine_id_project', dest='id_project',
                        help="The project from which we want the images")

    parser.add_argument('--size',
                        type=int,
                        default=300,
                        help="Size of the patch in pixels")

    parser.add_argument('--download_path', required=False,
                        default='./datasets/patchesa',
                        help="Where to store images")

    params, other = parser.parse_known_args(sys.argv[1:])

    id2info = {}
    with open('./cytomine/project-info.csv', 'r') as csvfile:

        f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        headers = next(f_csv)

        for row in f_csv:
            image_id = row[0]
            tissue = row[1]
            dye = row[2]

            id2info[image_id] = (tissue, dye)

    with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key,
                  verbose=logging.INFO) as cytomine:
        annotations = AnnotationCollection()
        annotations.project = params.id_project
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showGIS = True
        annotations.fetch()
        print(annotations)

        for annotation in annotations:
            print("ID: {} | Image: {} | Project: {} | Term: {} | User: {} | Area: {} | Perimeter: {} | WKT: {}".format(
                annotation.id,
                annotation.image,
                annotation.project,
                annotation.term,
                annotation.user,
                annotation.area,
                annotation.perimeter,
                annotation.location
            ))

            annot = Annotation().fetch(annotation.id)
            # Toutes les proprietes (collection) de l annotation
            properties = PropertyCollection(annot).fetch()
            # Une propriété avec une clé spécifique de l'annotation
            propert = Property(annot).fetch(key="ANNOTATION_GROUP_ID")

            image_id = str(annotation.image)

            if image_id in id2info:

                tissue, dye = id2info[image_id]

                path_patch = os.path.join(params.download_path, str(
                    params.size), tissue, dye, str(propert.value)+".jpg")

                if params.download_path and not os.path.exists(path_patch):
                    # default size is 300x300
                    annotation.dump(dest_pattern=path_patch,
                                    increase_area=params.size/100)
