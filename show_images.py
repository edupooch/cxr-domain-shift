import pandas as pd
import random
from eval_model import get_labels
import cv2

"""
Script to show image samples positive for a disease from one of the datasets
"""

DATASET_NAME = 'mimic'
FINDING = 'Consolidation'
PATH_TO_IMAGES = 'datasets/' + DATASET_NAME

df = pd.read_csv(DATASET_NAME + '_labels.csv')

for i in range(len(df['Image Index'])):
    if df[FINDING][i] > 0:
        print(df['Image Index'][i])
        img = cv2.imread(PATH_TO_IMAGES + '/' + df['Image Index'][i], 0)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
