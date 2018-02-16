# Requirements

* Tensorflow 1.5
* python 3.0>=

The `install.sh` file can be used to install the requirement

# Overview

The purpose of the project is to train a machine to sort picture into different categories.
The pictures, used to train, are ine the train folder.
The labels to categories the picture are the folder names.
After the training, the script `photoSort.py` will use the pictures inside the folder `test`
and will generate a `result.csv` containing the name and the category of each picture.

A `result.csv` file is already in the folder to present previous calculated resluts

#Usage

Use the train.py to generate the files used by **photoSort.py**

`$ python3 train.py`

`$ python3 photoSort.py`

`$ cat result.csv`
