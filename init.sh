#!/bin/bash

mkdir datasets
mkdir results

user="ANHIR-guest"
password="isbi2019"

echo "Fetching landmarks.."

wget http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/landmarks/dataset_medium.zip --user "$user" --password "$password"
unzip dataset_medium.zip -d datasets/landmarks
rm dataset_medium.zip 

echo "Fetching images.."

wget http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/images/dataset_medium.zip --user "$user" --password "$password"
wget http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/images/dataset_medium.z01 --user "$user" --password "$password"
wget http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/images/dataset_medium.z02 --user "$user" --password "$password"
wget http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/images/dataset_medium.z03 --user "$user" --password "$password"
wget http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/images/dataset_medium.z04 --user "$user" --password "$password"
wget http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/images/dataset_medium.z05 --user "$user" --password "$password"

unzip dataset_medium.z\* -d datasets/images 
rm dataset_medium.z*

echo "Setting up.."

pip3 install -r requirements.txt
python3 setup.py develop
python3 info/get_info_locally.py 
python3 datasets/normalization.py
python3 datasets/generate_dataset.py