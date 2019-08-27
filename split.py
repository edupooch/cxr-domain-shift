import pandas as pd
import random

"""
Script to re-split (train, validation, test) chexpert and mimic dataset
"""

DATASET_NAME = 'chexpert'

df = pd.read_csv(DATASET_NAME + '/original/' + DATASET_NAME + '_labels.csv')
patient_dict = {}

for i in range(len(df['Image Index'])):
    if DATASET_NAME == 'chexpert':
        patient = df['Image Index'][i].split('/')[2]
    if DATASET_NAME == 'mimic':
        patient = df['Image Index'][i].split('/')[1]

    if patient in patient_dict:
        patient_dict.update({patient: [i] + patient_dict[patient]})
    else:
        patient_dict.update({patient: [i]})

key_list = list(patient_dict.keys())
total_patients = len(key_list)
print(total_patients)

random.shuffle(key_list)
train = key_list[0: int(total_patients*0.7)]
val = key_list[int(total_patients*0.7): int(total_patients*0.8)]
test = key_list[int(total_patients*0.8): int(total_patients)]

train_ids = []
for key in train:
    train_ids = train_ids + patient_dict[key]

val_ids = []
for key in val:
    val_ids = val_ids + patient_dict[key]

test_ids = []
for key in test:
    test_ids = test_ids + patient_dict[key]

df['fold'].iloc[train_ids] = 'train'
df['fold'].iloc[val_ids] = 'val'
df['fold'].iloc[test_ids] = 'test'

df.to_csv('splits.csv')
