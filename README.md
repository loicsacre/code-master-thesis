# Code of Loïc Sacré's master thesis

:exclamation: 
    - The code works with Python3 only
    - Call all the processes from the root of the project
    - Require the Cytomine client and utilities in Python (https://github.com/cytomine/Cytomine-python-client)

## Init

- run ./init.sh
- Note that all the main paths for the data are stored in 'path/path' in the Paths class. It is not recommended to change them. They are shared through the whole project. By default:
    - the path to the images have to be stored in "./datasets/images/"
    - the path to the landmarks have to be stored in "./datasets/landmarks/"

or follow all these steps:

- Run the setup file (with 'python3 setup.py install')
- Get the images and landmarks and put them in the folder './datasets' with the previously defined paths:
    - url for everything: https://anhir.grand-challenge.org/Download/
- (Optional: Upload the images to a Cytomine project)
    - :exclamation: if the images are uploaded to a project, please use the following format: 'tissue&scale&original_name' 
        - The images from  https://anhir.grand-challenge.org/Download/ are stored like : tissue/scale/original_name.ext (where ext is either jpg or png)
        - My actual project has the names in the format 'tissue_original_name' (which is not great) because 'some tissue' and 'original_name' contain the '_' character

- Setup and install requirements with 'setup.py' and 'requirements.txt'
- Generate the info CSV either locally (with info/get_info_locally.py) or from a project (with info/get_info_from_project.py)
    - The file can be easily adapted but consider to keep the same order in the fieldnames
- Generate the normalization info with './datasets/normalization.py'

## Process Dataset for Training

- Get all the needed patches locally ('info/get_patches_locally.py') or from a project ('info/get_patches_from_project.py')
    - Get patches locally
    - They will be stored automatically in 'datasets/patches/{size}' (where size is the size of the patches)
        - Note that you can change the path in 'path/path' (or give in as an argument, not recommended)
- Generate the dataset by running the file 'generate_dataset.py' (specifying the size)
- For the Alan-cluster: Put the patches in the scratch folder to speed-up the training

## ImageNet

### Experiment 1 

see './imagenet/1/'

### Experiment 2

see './imagenet/2/'

## MatchNet and Siamese Architecture

All experiments with the MatchNet-like (resp. Siamese) network architectures are in the folder 'matchnet' (resp. 'siamese')
