#!/bin/bash

# Connect with ssh to a node and copy the dataset on which you would like to train

size='300'

mkdir -p /scratch/"$USER"/datasets
mkdir /scratch/"$USER"/datasets/patches
scp -r alan-master:datasets/patches/"$size" /scratch/"$USER"/datasets/patches/ 

