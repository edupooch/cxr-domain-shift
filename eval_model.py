import torch
import pandas as pd
from loader import get_loader
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np


def get_labels(DATASET_NAME):
    if DATASET_NAME == 'nih':
        return [
            'No Finding',
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Lung Lesion',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia'
        ]
    if DATASET_NAME == 'chexpert' or DATASET_NAME == 'mimic':
        return [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']
    if DATASET_NAME == 'common':
        return [
            'No Finding',
            'Atelectasis',
            'Cardiomegaly',
            'Lung Lesion',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema']


def eval_model(data_transforms, model, PATH_TO_IMAGES, TRAIN_DATASET):
    """
    Evaluates model on the source dataset after training
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which test images can be found
        TRAIN_DATASET: the name of the dataset in which the model was trained (nih, chexpert or mimic)
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # calc preds in batches of 32, can reduce if your GPU has less RAM
    BATCH_SIZE = 32

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    CXR = get_loader(TEST_DATASET)
    # create dataloader
    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        transform=data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(dataset)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        if(i % 10 == 0):
            print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calc AUCs
    for column in true_df:

        if column not in get_labels(TRAIN_DATASET):
            continue

        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(
                actual.as_matrix().astype(int), pred.as_matrix())
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)

    pred_df.to_csv(TRAIN_DATASET + "/results/preds_" +
                   TRAIN_DATASET + ".csv", index=False)
    auc_df.to_csv(TRAIN_DATASET + "/results/aucs_" +
                  TRAIN_DATASET + ".csv", index=False)
    return pred_df, auc_df


def make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES, TRAIN_DATASET, TEST_DATASET):
    """
    Gives predictions for test fold using previously trained model and any of the three test datasets

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which test images can be found
        TRAIN_DATASET: the name of the dataset in which the model was trained (nih, chexpert or mimic)
        TEST_DATASET: the name of the dataset to evaluate the model performance (nih, chexpert or mimic)
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
    """

    # calc preds in batches of 32, can reduce if your GPU has less RAM
    BATCH_SIZE = 32

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    CXR = get_loader(TEST_DATASET)

    # create dataloader
    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        transform=data_transforms['val'])

    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(dataset)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    TRAIN_LABELS = get_labels(TRAIN_DATASET)

    # iterate over dataloader
    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        batch_size = labels.cpu().data.numpy().shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(TRAIN_LABELS)):
                thisrow["prob_" + TRAIN_LABELS[k]] = probs[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)

        if(i % 10 == 0):
            print(str(i * BATCH_SIZE))

    pred_df.to_csv(TEST_DATASET + "/results/preds_" +
                   TRAIN_DATASET + ".csv", index=False)

    return pred_df


def calc_aucs(pred_df, TRAIN_DATASET, TEST_DATASET):
    """
    Compute AUCS given the model predictions

    Args:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        TRAIN_DATASET: the name of the dataset in which the model was trained (nih, chexpert or mimic)
        TEST_DATASET: the name of the dataset to evaluate the model performance (nih, chexpert or mimic)
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
    """
    
    
    df = pd.read_csv(TEST_DATASET + '_labels.csv')
    true_df = df[df['Image Index'].isin(pred_df['Image Index'])]
    # print(df.shape)
    # print(true_df.shape)
    # calc AUCs
    auc_df = pd.DataFrame(columns=["label", "auc"])
    for column in true_df:

        if column not in get_labels('common'):
            continue

        actual = true_df[column]
        actual.replace(
            to_replace=[-1.0],
            value=0,
            inplace=True
        )
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:

            thisrow['auc'] = sklm.roc_auc_score(
                actual.fillna(0).as_matrix().astype(int), pred.as_matrix())

        except BaseException as e:
            print("can't calculate auc for " + str(column))
            print(str(e))
        auc_df = auc_df.append(thisrow, ignore_index=True)

    auc_df.to_csv(TEST_DATASET + "/results/aucs_" +
                  TRAIN_DATASET + ".csv", index=False)
