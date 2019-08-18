
import csv
import os

from PIL import Image

with open('./info/project-info.csv', 'r') as csvfile:

    f_csv = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
    next(f_csv)

    for row in f_csv:

        image_id = row[0]
        tissue = row[1]
        dye = row[2]
        height = row[3]
        width = row[4]
        nb_of_landmarks = int(row[5])
        original_name = row[6]
        scale = row[7]
