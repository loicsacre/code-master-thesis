#!/bin/bash

echo "Removing .DS_Store files";
find . -name '.DS_Store' -type f -delete;
echo "Removing ._* files";
find . -type f -name '._*' -delete;


echo "Install setup"
python3 setup.py clean --all
python3 setup.py install --record files.txt
cat files.txt | xargs rm -rf
rm files.txt
rm -r build
python3 setup.py develop
rm -r loicsacre_master_thesis.egg-info
rm -r dist
rm -r .ipynb_checkpoints
