import pandas as pd
import random
from eval_model import get_labels

"""
Script to compute the frequency of positive labels in each dataset
"""

DATASET_NAME = 'chexpert'

df = pd.read_csv(DATASET_NAME + '_labels.csv')
label_names = get_labels('common')

label_freq_train = [0 for i in range(len(label_names))]
label_freq_test = [0 for i in range(len(label_names))]
print(label_names)
for i in range(len(df['Image Index'])):
    for j in range(len(label_names)):
        if df[label_names[j]][i] > 0:
            if df['fold'][i] == 'train':
                label_freq_train[j] += 1
            elif df['fold'][i] == 'test':
                label_freq_test[j] += 1
    
print(label_freq_train)
print(label_freq_test)