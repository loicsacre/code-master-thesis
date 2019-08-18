# -*- coding: utf-8 -*-

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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import logging
import os
import sys
from argparse import ArgumentParser

from shapely.geometry import Point, box

from cytomine import Cytomine
from cytomine.models import (Annotation, AnnotationCollection, AnnotationTerm,
                             Property)
from path import Paths

__author__ = "Rubens Ulysse <urubens@uliege.be>"  # modified by Loic Sacre

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
    parser.add_argument('--cytomine_id_image_instance', dest='id_image_instance',
                        help="The image to which the annotation will be added")
    parser.add_argument('--cytomine_id_term', dest='id_term', required=False,
                        help="The term to associate to the annotations (optional)")
    parser.add_argument('--landmarks', dest='landmarks',
                        default=Paths.PATH_TO_LANDMARKS,
                        help="The path where the annotations are stored")

    params, _ = parser.parse_known_args(sys.argv[1:])

    with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key,
                  verbose=logging.INFO) as cytomine:

        with open('./info/project-info.csv', 'r') as csvfile:

            f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
            next(f_csv)

            for row in f_csv:

                image_id = row[0]
                tissue = row[1]
                height = float(row[3])
                original_name = row[6]
                scale = row[7]

                path_to_landmarks = os.path.join(
                    params.landmarks, tissue, scale, f"{original_name}.csv")

                with open(path_to_landmarks, 'r') as csvfile:

                    f_csv = csv.reader(
                        csvfile, delimiter=str(','), quotechar=str('|'))
                    headers = next(f_csv)
                    annotations = AnnotationCollection()

                    for row in f_csv:

                        id_landmark = int(row[0])

                        # due to Cytomine
                        point = Point(float(row[1]), height - float(row[2]))

                        a = Annotation(
                            location=point.wkt, id_image=image_id, id_project=params.id_project)
                        a.property = [
                            {"key": "ANNOTATION_GROUP_ID", "value": id_landmark}]
                        annotations.append(a)

                    annotations.save()
