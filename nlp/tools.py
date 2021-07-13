import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score



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
    return {"train": train_folds, "test": test_fold}


def evaluate(y_true, y_probas):
    """
    Evaluates the prediction-probabilities of a model
    using accuracy and roc-auc score

    :param torch.Tensor y_true: true labels
    :param torch.Tensor y_probas: predicted class probabilities
    :return: a dictionary containing the accuracy and roc-auc score
    """
    probas_np = y_probas.cpu().detach().numpy()
    preds_np = np.round(y_probas.cpu().detach().numpy())
    y_batch_np = y_true.cpu().detach().numpy()
    acc = accuracy_score(y_true=y_batch_np, y_pred=preds_np)
    try:
        roc_auc = roc_auc_score(y_true=y_batch_np, y_score=probas_np)
    except ValueError:
        roc_auc = 0.5
    return {"acc": acc, "roc_auc": roc_auc}


def train_val_split(data_folds, val_fold_id):
    """
    Concatenates a number of training folds and provides a single validation fold.

    :param list data_folds: list of data folds, data folds are pandas.DataFrames
    :param int val_fold_id: index of the validation fold in data_folds
    :return: dictionary containing train and validation data
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


def parameters_rnn_based(n_epochs, lr, max_seq_len, n_layers, feats_per_time_step, hidden_size, batch_size,
                         device, n_classes=2, x_name="text", y_name="label"):
    """
    Creates a dictionary containing the necessary preprocessing, model and training parameters for all wrappers
    based on recurrent architectures. (RNNWrapper, EmbeddingWrapper, GloveWrapper)

    :param int n_epochs: maximum number of epochs
    :param float lr: initial learning rate
    :param int max_seq_len: sequence length to which all sequences have to be padded / truncated
    :param int n_layers: number of RNN / GRU / LSTM layers
    :param int feats_per_time_step: number of predictors per timestep. In case of word embeddings, this reflects the
    embedding size per timestep / word
    :param int hidden_size: size of the hidden state
    :param int batch_size: number of observations per batch
    :param str device: name of the utilized device (either cpu or cuda)
    :param int n_classes: number of classes. 2 in a binary classification task
    :param str x_name: name of the column containing the textual information
    :param str y_name: name of the column containing the labels
    :return: a dictionary containing all parameters having their names as keys.
    """
    return {"n_epochs": n_epochs, "lr": lr, "max_seq_len": max_seq_len, "n_layers": n_layers,
            "feats_per_time_step": feats_per_time_step, "hidden_size": hidden_size, "batch_size": batch_size,
            "device": device, "n_classes": n_classes, "x_name": x_name, "y_name": y_name}


def parameters_bert_based(n_epochs, lr, max_seq_len, batch_size, device, n_classes=2, x_name="text", y_name="label"):
    """
    Creates a dictionary containing the necessary preprocessing, model and training parameters for the BertWrapper.

    :param int n_epochs: maximum number of epochs
    :param float lr: initial learning rate
    :param int max_seq_len: sequence length to which all sequences have to be padded / truncated
    :param int batch_size: number of observations per batch
    :param str device: name of the utilized device (either cpu or cuda)
    :param int n_classes: number of classes. 2 in a binary classification task
    :param str x_name: name of the column containing the textual information
    :param str y_name: name of the column containing the labels
    :return: a dictionary containing all parameters having their names as keys.
    """
    return {"n_epochs": n_epochs, "lr": lr, "max_seq_len": max_seq_len, "batch_size": batch_size,
            "device": device, "n_classes": n_classes, "x_name": x_name, "y_name": y_name}


def performance_comparison(parameter_combinations, wrapper, folds, model_name):
    """
    Compares the performance of the models embedded in >wrapper< and visualizes the results in .png files.

    :param list parameter_combinations: a list of parameter combinations used by the model.
    :param wrapper: a model-wrapper
    :param pd.DataFrame folds: the concatenated data folds on which the model parameters have to be evaluated
    :param str model_name: name of the model
    """
    for i, parameters in enumerate(parameter_combinations):
        metrics = wrapper.evaluate_hyperparameters(folds=folds, parameters=parameters)
        acc_scores_train = pd.Series(metrics["acc_scores_train"], name="Train Accuracy")
        roc_auc_scores_train = pd.Series(metrics["roc_auc_scores_train"], name="Train ROC-AUC")

        acc_scores = pd.Series(metrics["acc_scores"], name="Validation Accuracy")
        roc_auc_scores = pd.Series(metrics["roc_auc_scores"], name="Validation ROC-AUC")


        # plot
        fig, axs = plt.subplots(2, figsize=(5, 10))
        fig.suptitle(f"{model_name}")
        x_labels = range(1, len(acc_scores) + 1)

        acc_scores_train.plot(ax=axs[0], c="red", ls=("dashed"))
        roc_auc_scores_train.plot(ax=axs[1], c="blue", ls=("dashed"))

        acc_scores.plot(ax=axs[0], c="red")
        roc_auc_scores.plot(ax=axs[1], c="blue")

        axs[0].set_title("Accuracy Score")
        axs[0].legend()
        min = np.min([acc_scores_train.min(), acc_scores.min()])
        max = np.max([acc_scores_train.max(), acc_scores.max()])
        axs[0].set_ylim([min, max])
        axs[0].set_xticks(range(len(acc_scores)))
        axs[0].set_xticklabels(x_labels)
        axs[0].set_xlabel("Epochs")

        axs[1].set_title("ROC-AUC Score")
        axs[1].legend()
        min = np.min([roc_auc_scores_train.min(), roc_auc_scores.min()])
        max = np.max([roc_auc_scores_train.max(), roc_auc_scores.max()])
        axs[1].set_ylim([min, max])
        axs[1].set_xticks(range(len(acc_scores)))
        axs[1].set_xticklabels(x_labels)
        axs[1].set_xlabel("Epochs")

        plt.tight_layout(pad=3)
        plt.savefig("visuals/" + model_name + "_combi_" + str(i + 1))
