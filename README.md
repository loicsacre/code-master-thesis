# Loïc Sacré's Code for the Master Thesis


![](https://uliege.cytomine.org/images2/logo-300-ulg.png)

## :exclamation: Before going further :exclamation:

- The code works with Python3 only and uses the Pytorch framework

- Call all the processes from the root of the project

- Require the [**Cytomine** client and utilities in Python](https://github.com/cytomine/Cytomine-python-client) if a project is created on the Cytomine software.

- Every implementations are written so that the processes run locally to avoid a useless overhead communication with the servers of Cytomine. However the codes can easily be adapted.

- For using the [Alan GPU cluster](https://github.com/montefiore-ai/alan-cluster), some templates are available in `./alan_template_scripts`


## Init

Two options offer to you:
  
- _**Option 1:** set the project automatically_
    - run `./init.sh`

        - Note that all the main paths for the data are stored in `path/path` in the Paths class. It is not recommended to change them. They are shared through the whole project. By default:

         - the images have to be stored in `./datasets/images/`

         - the landmarks have to be stored in `./datasets/landmarks/`

 
- _**Option 2:** step by step initialization_


    - Run the setup file (with `python python3 setup.py install`)

    - Get the images and landmarks and put them in the folder `./datasets` with the previously defined paths:

        - url for everything: https://anhir.grand-challenge.org/Download/

    - (Optional: Upload the images to a **Cytomine** project)

         - :exclamation: If the images are uploaded to a project, please use the following format: 'tissue&scale&original_name'

         - The images from https://anhir.grand-challenge.org/Download/ are stored like: 'tissue/scale/original_name.ext' (where ext is either _jpg_ or _png_)

         - My actual project has the names in the format 'tissue_original_name' (which is not great) because some 'tissue' and 'original_name' contain the '_' character


    - Setup and install requirements with `setup.py` and `requirements.txt`

    - Generate the info CSV either locally (with `./info/get_info_locally.py`) or from a project (with `./info/get_info_from_project.py`)

         - The file can be easily adapted but consider to keep the same order in the fieldnames

    - Generate the normalization info with `./datasets/normalization.py`

  
## Approach 1: Transfer Learning via ImageNet Siamese Networks

### Experiment 1: Evaluation of ImageNet Networks


The code is in the `./imagenet/1/` folder. The code is organized as follows. There are four files. `main.py` is the main program for generating an evalutation for a certain pre-trained network. It consists of 3 phases:
- Getting the features from all patches coming from the landmarks(`features.py`)
- Generating the similarity measures for the patch pairs (`compare.py`)
- Generating the top-k accuracy (for k=1 and 5) (`accuracy.py`)

Usage examples: 
- `python3 imagenet/1/main.py --arch densenet161 --size 600`
- `python3 imagenet/1/main.py --arch resnet50 --size 600 -pool` (pooling is activated)


### Experiment 2: : Evaluation on Segmented Whole-Slide Images

The code is in the `./imagenet/2/` folder. The main program is in `main.py`. The several phases consists in:
- Getting the patches from the reference image (landmarks)
- Getting the patches from the target image (segmented)
- Compare thanks to the a certain distance (by default cosine similarity)
- Computing top-k accuracy (for k=1 and 5) 

The code follows the same procedure described in the corresponding section of the report


Usage examples:
- `python3 imagenet/2/main.py --arch densenet201--size 300 --shift 75`
- `python3 imagenet/2/main.py --arch densenet201 --size 600 --shift 150 --resize 300`

#### Visualize Heat Map for Experiment 2

- `visualize-results-imagenet`: Generate an heat map for a certain pair. The dye1 is the reference and dye2 the target:
    - Usage: `python3 python3 visualize/visualize-results-imagenet.py --tissue lung-lobes_2 --dye1 CD31 --dye2 CC10 --arch densenet201 --size 300 --references 3 7 --data 'filename'`
    - 'filename' is an output file generated wih `imagenet/2/main.py`


- `visualize-segmentation.py`: generates the segmentation of all images in dataset in order to visualize them
    - Usage: `python3 visualize-segmentation.py --size 300 --shift 75`

  
## Approach 2: Models Trained on the Data

Each of folders `./matchnet/` and `./siamese/`
contains `main.py` and `dataset_+++.py` files and and a model `folder` 

- `main.py` is the main program for training and testing the networks
- `dataset_+++.py` contains the dataloader for the training
- `model`contains all the models architecture for a 


### Process Dataset for Training

- Get all the needed patches locally (`./info/get_patches_locally.py`) or from a project (`./info/get_patches_from_project.py`)

     - They will be stored automatically in `./datasets/patches/{size}` (where size is the size of the patches)

 - Note that you can change the path in `./path/path` (or give in as an argument, not recommended)

- Generate the dataset by running the file `generate_dataset.py` (specifying the size). It generates the split of the dataset into training, evaluation and testing in the file `./datasets/cnn/dataset-networks.data`

- For the [Alan GPU cluster]([https://github.com/montefiore-ai/alan-cluster](https://github.com/montefiore-ai/alan-cluster)
): put the patches in the scratch folder to speed-up the training (with the use of `copy.sh`)

### MatchNet and TransferNet

The code is in the `./matchnet/` folder.

Usage examples:
- `python3 matchnet/main.py --arch matchnet --learning_rate 0.005 --momentum 0.9 --batch_size 64`
- `python3 matchnet/main.py --arch transferAlexnet --learning_rate 0.001 --momentum 0.9 --batch_size 64`


### Siamese-like 

The code is in the `./siamese/` folder.

Usage examples:
- `python3 matchnet/main.py --arch siameseAlexnet --learning_rate 0.01  --momentum 0.9 --batch_size 64 --margin 0.5`

## Comparing both Approches on the Testing Set

The code is in the folder `./evaluate`.

### Evaluating the Approach 1

The code is in the file `evaluate_imagenet.py`.  It generates the total top-k accuracy (for k=1 and 5)  from a CSV file generated thanks to `imagenet/2/summary.py` for a specific pre-trained network.



### Evaluating the Approach 2

The code is in the file `evaluate.py`. Compute the top-k accuracy  (for k=1 and 5) for a generated model if the checkpoint file has been generated.

Usage:
- `python3 ./evaluate/evaluate.py --arch siameseAlexnet --size 300 --shift 75 --checkpoint ./results/siamese/siameseAlexnet/checkpoints/siameseAlexnet_0.01_0.0_64-1219943.check`


## Utils

Here are listed all the _utils_ functions. These functions are shared among the project to avoid uncessary repeated code. The files are in the folder `./utils`. 

- `Normalizer.py`: contains a Normalizer class to perform per image normalization. Please consider to generate the file containing all the info with `./datasets/normalization.py`.
- `FeaturesExtractor.py`: contains a FeaturesExtractor class. It is responsible for extracting the feature vector for a certain pre-trained network (Only useful for the Approach 1).
- `utils.py`: contains a compare function to compare two feature vectors with a certain distance.
- `segmentation.py`: contains all functions related to the management of the images, segmenting an image or getting the patches from an image (easily adaptable to the conventions of Cytomine, i.e. the system of coordinates used in for an image)
- `visualization.py`: contains all functions  related to the heat maps generation.




