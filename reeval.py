import eval_model as E

import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from model import get_data_transforms, datasets

"""
Script to re-evaluate a model trained in one dataset.

Args:
 The dataset in which the model was trained ('nih', 'chexpert' or 'mimic')
 The dataset to test the model.  ('nih', 'chexpert' or 'mimic')
"""

DATASET_NAMES = ('nih', 'chexpert', 'mimic')

if len(sys.argv) != 3:
    raise Exception('Two argument expected (train and test dataset)')

TRAIN_DATASET = sys.argv[1]
TEST_DATASET = sys.argv[2]

if (TRAIN_DATASET not in DATASET_NAMES) or (TEST_DATASET not in DATASET_NAMES):
    raise Exception('Invalid dataset name, needs to be one of the following: nih, chexpert, mimic')

# torchvision transforms
data_transforms = get_data_transforms()

PATH_TO_MODEL = '/A/eduardo/projects/cxr_gen/' + TRAIN_DATASET + '/model/modelinf.pt'
PATH_TO_IMAGES = '/A/eduardo/datasets/' + TEST_DATASET

N_LABELS = 14

# load model
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
# add final layer with # outputs in same dimension of labels with sigmoid
# activation
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

model.load_state_dict(torch.load(PATH_TO_MODEL))
# put model on GPU
model = model.cuda()


preds = E.make_pred_multilabel(
    data_transforms, model, PATH_TO_IMAGES, TRAIN_DATASET, TEST_DATASET)
E.calc_aucs(preds, TRAIN_DATASET, TEST_DATASET)
