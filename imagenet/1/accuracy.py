
import numpy as np


def get_accuracy(data, distance="cos"):

    top1accuracy = 0
    top5accuracy = 0

    k_list = [1, 5]
    # count the landmarks respecting the condition
    counter = [0]*len(k_list)

    for k_ind, k in enumerate(k_list):
        total = 0
        for tissue in data:
            for pair in data[tissue]:
                nb_annot = data[tissue][pair].shape[0]

                # (0) for the 1st dye compared to the 2nd,
                # (1) for the 2nd compared to the 1st
                for r in range(2):
                    total += nb_annot
                    for i in range(nb_annot):
                        array = data[tissue][pair]
                        if r:
                            # Vertically
                            array = [(x, i) for i, x in enumerate(array[:, i])]
                        else:
                            # Horizontally
                            array = [(x, i) for i, x in enumerate(array[i, :])]

                        array.sort(key=lambda x: x[0])

                        if distance == "cos" and i in [x[1] for x in array[-k:]]:
                            counter[k_ind] += 1
                        if (distance == "eucl" or distance == "eucl-norm") and i in [x[1] for x in array[:k]]:
                            counter[k_ind] += 1

        if k == 1:
            top1accuracy = round(counter[0]/total, 4)
        else:
            top5accuracy = round(counter[1]/total, 4)

    return top1accuracy, top5accuracy
