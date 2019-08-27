import sys
import eval_model as E
import model as M

"""
Script to retrain the model

Args:
 Train dataset name ('nih', 'chexpert' or 'mimic')
"""

DATASET_NAMES = ('nih', 'chexpert', 'mimic')

if len(sys.argv) != 2:
    raise Exception('One argument expected (dataset name)')

DATASET_NAME = sys.argv[1]

if DATASET_NAME not in DATASET_NAMES:
    raise Exception('Invalid dataset name, needs to be one of the following: nih, chexpert, mimic')

PATH_TO_IMAGES = 'datasets/' + DATASET_NAME

WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE,
                          WEIGHT_DECAY, DATASET_NAME=DATASET_NAME)
