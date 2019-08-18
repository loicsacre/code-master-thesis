#!/bin/bash

# Connect with ssh to a node and copy the dataset with which you would like to train

size='300'
username='lsacre'

mkdir /scratch/lsacre/datasets
mkdir /scratch/lsacre/datasets/patches
scp -r alan-master:datasets/patches/"$size" /scratch/lsacre/datasets/patches/ 

