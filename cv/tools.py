import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def select_device():
    """
    Tries to detect a Cuda-GPU.
    Detects the CPU if no GPU available.

    :return: the name of the detected device.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # clean GPU
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def read_folds(prefix, read_path, num_folds=6, test_fold_id=0):
    """
    Reads num_folds many folds of data having the same prefix.
    Each fold is stored as a csv file.

    :param str prefix: the prefix of the folds (e.g. <text_fold>)
    :param str read_path: the path of the directory that contains the folds
    :param int num_folds: number of folds to read
    :param int test_fold_id: the index of the fold containing test data
    :return: dictionary containing all read folds. Folds are represented as pd.DataFrames.
    Folds used for training and train-validation are stored in "train", testing data is stored in "test".
    """
    train_folds = []
    test_fold = None
    for i in range(num_folds):
        if i == test_fold_id:
            test_fold = pd.read_csv(read_path + "/" + prefix + str(i) + ".csv")
        else:
            train_folds.append(pd.read_csv(read_path + "/" + prefix + str(i) + ".csv"))
    return {"available_for_train": train_folds, "test": test_fold}


def evaluate(y_true, y_probas):
    """
    Evaluates the prediction-probabilities of a model
    using the accuracy_score and the f1_score.

    :param torch.Tensor y_true: true labels
    :param torch.Tensor y_probas: predicted class probabilities
    :return: dict, a dictionary having the keys "acc", and "f1" and the respective values.
    """
    preds_batch_np = np.round(y_probas.cpu().detach().numpy())
    y_batch_np = y_true.cpu().detach().numpy()
    acc = accuracy_score(y_true=y_batch_np, y_pred=preds_batch_np)
    f1 = f1_score(y_true=y_batch_np, y_pred=preds_batch_np, average='weighted')
    return {"acc": acc, "f1": f1}


def train_val_split(data_folds, val_fold_id):
    """
    Given all folds, concatenates all training folds.
    Returns training, validation and testing data.

    :param list data_folds: list of data folds, data folds ara pandas.DataFrames with columns "text" and "label"
    :param int val_fold_id: index of the validation fold in data_folds
    :return: dictionary containing keys X_train, y_train, X_val, y_val, X_test, y_test,
    respective values are pd.Series objects
    """
    train_data = None
    initiated = False
    for i, fold in enumerate(data_folds):
        if i != val_fold_id:
            if not initiated:
                train_data = data_folds[i]
                initiated = True
            else:
                train_data = pd.concat([train_data, data_folds[i]], ignore_index=True)
    val_data = data_folds[val_fold_id]
    return {"train": train_data, "val": val_data}
