# CXR domain shift
This repository provides the code for reproducing the experiments of our paper "[Can we trust deep learning models diagnosis? The impact of domain shift in chest radiograph classification.](https://arxiv.org/abs/1909.01940)"

## Data
The data used in this repository comes from three different datasets, [NIH ChestX-ray14](https://arxiv.org/abs/1705.02315), [CheXpert](https://arxiv.org/abs/1901.07031), and [MIMIC-CXR](https://arxiv.org/abs/1901.07042). The three datasets are publicly available. We provide pre-trained models on all three datasets. If you want to retrain the models you need to download the datasets from their sources.

## Experiments
We train three models, one for each dataset, and subsequently evaluate our model at the other two. Each model is trained with the training set and evaluated at the other two test sets. The three datasets have the same train, test and validation sets in all experiments. 

Inside each dataset's directory there is a directory with the model trained on it along with a checkpoint and training log. Also, the results obtained by the model on the test set of each of the three datasets. Inside the *results* directory, there is the prediction score for the each sample of the test set and AUCs for each disease (common between the train and test set).

To retrain a model see *retrain.py* and to re-evaluate a trained model see *reeval.py*.

## Acknowledgments
Special thanks to [jrzech](https://github.com/jrzech) for his code reproducing CheXnet in Pytorch, in which we based our experiments. Original code [available on GitHub](https://github.com/jrzech/reproduce-chexnet).
