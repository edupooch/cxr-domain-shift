from model import datasets
import chexpert.loader as chexpert_loader
import nih.loader as nih_loader
import mimic.loader as mimic_loader


def get_loader(DATASET_NAME):
    if DATASET_NAME == 'nih':
        return nih_loader
    if  DATASET_NAME == 'chexpert':
        return chexpert_loader
    if  DATASET_NAME == 'mimic':
        return mimic_loader
