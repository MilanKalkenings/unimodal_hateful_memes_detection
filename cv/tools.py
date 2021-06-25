import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.utils.data import Dataset


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
    Folds are stored in "train", or "test" sets. Each of the former strings are keys of the
    returned dictionary.
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
    using accuracy, precision, and recall score

    :param torch.Tensor y_true: true labels
    :param torch.Tensor y_probas: predicted class probabilities
    :return: a dictionary containing the accuracy, precision, and recall score
    """
    preds_batch_np = np.round(y_probas.cpu().detach().numpy())
    y_batch_np = y_true.cpu().detach().numpy()
    acc = accuracy_score(y_true=y_batch_np, y_pred=preds_batch_np)
    precision = precision_score(y_true=y_batch_np, y_pred=preds_batch_np, zero_division=1)
    recall = recall_score(y_true=y_batch_np, y_pred=preds_batch_np, zero_division=1)
    return {"acc": acc, "precision": precision, "recall": recall}


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


class CustomDataset(Dataset):
    """
    A custom Image Dataset that performs transformations on the images contained in it and shifts them to
    a given device.
    """

    def __init__(self, data, transform_pipe, x_name="img", y_name="label", device="cuda"):
        """
        Constructor.

        :param pd.DataFrame data: A DataFrame containing one column of image paths and another columns of image labels.
        :param transform_pipe: a transform:Composition of all transformations that have to be applied to the images
        :param str x_name: name of the image column
        :param str y_name: name of the label column
        :param str device: name of the device that has to be used
        """
        self.data = data
        self.transform_pipe = transform_pipe
        self.x_name = x_name
        self.y_name = y_name
        self.device = device

    def __len__(self):
        """
        Returns the number of observations in the whole dataset

        :return: the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, i):
        """
        Is used by DataLoaders to draw the observation at index i in the dataset.

        :param int i: index of an observation
        :return: a list containing the image-data and the label of one observation
        """
        img_path = "../../data/hateful_memes_data/" + self.data[self.x_name].iloc[i]
        x = self.transform_pipe(Image.open(img_path, formats=["PNG"])).to(self.device)
        if x.size(0) == 4:  # very few images have one more channel, change to RGB format
            image = Image.open(img_path, formats=["PNG"]).convert("RGB")
            x = self.transform_pipe(image).to(self.device)
        y = torch.tensor(self.data[self.y_name][i], dtype=torch.float).to(self.device)
        return [x, y]


def parameters_cnn(n_epochs, lr, batch_size, transform_pipe, conv_ch1, conv_ch2, linear_size, kernel_size,
                   pooling_size, accumulation, device):
    """
    Creates a dictionary containing the necessary preprocessing, model and training parameters for the CNNWrapper.

    :param int n_epochs: maximum number of epochs
    :param float lr: initial learning rate
    :param batch_size: number of observations per batch
    :param int accumulation: number of batches accumulated to form a single gradient per parameter
    :param transform_pipe: a pipeline consisting of image transformations. Should at least ensure the images to have
    a symetric shape and being stored in torch.Tensors
    :param int conv_ch1: number of channels after applying the fist convolution
    :param int conv_ch2: number of channels after applying the second convolution
    :param int linear_size: size of the second linear layer. Size of the first linear layer is determined automatically
    :param int kernel_size: width and height of the convolutional kernels / filters / windows.
    :param int pooling_size: width and height of the maximum pooling window
    :param str device: name of the utilized device (either cpu or cuda)
    :return: a dictionary containing all parameters having their names as keys.
    """
    return {"n_epochs": n_epochs, "lr": lr, "batch_size": batch_size, "transform_pipe": transform_pipe,
            "conv_ch1": conv_ch1, "conv_ch2": conv_ch2, "linear_size": linear_size, "kernel_size": kernel_size,
            "pooling_size": pooling_size, "accumulation": accumulation, "device": device}


def parameters_pretrained(n_epochs, lr, batch_size, transform_pipe, pretrained_component, linear_size, freeze_epochs,
                          unfreeze_epochs, device):
    """
    Creates a dictionary containing the necessary preprocessing,
    model and training parameters for the PretrainedWrapper.

    :param int n_epochs: maximum number of epochs
    :param float lr: initial learning rate
    :param batch_size: number of observations per batch
    :param transform_pipe: a pipeline consisting of image transformations. Should at least ensure the images to have
    a symetric shape and being stored in torch.Tensors
    :param pretrained_component:
    :param int linear_size: size of the second linear layer. Size of the first linear layer is determined automatically
    :param list freeze_epochs: a list of integers representing the epochs in which the pretrained component
    has to be frozen
    :param list unfreeze_epochs: a list of integers representing the epochs in which the pretrained component
    has to be unfrozen
    :param str device: name of the utilized device (either cpu or cuda)
    :return: a dictionary containing all parameters having their names as keys.
    """
    return {"n_epochs": n_epochs, "lr": lr, "batch_size": batch_size, "transform_pipe": transform_pipe,
            "pretrained_component": pretrained_component, "linear_size": linear_size, "freeze_epochs": freeze_epochs,
            "unfreeze_epochs": unfreeze_epochs, "device": device}


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
        precision_scores_train = pd.Series(metrics["precision_scores_train"], name="Train Precision")
        recall_scores_train = pd.Series(metrics["recall_scores_train"], name="Train Recall")

        acc_scores = pd.Series(metrics["acc_scores"], name="Validation Accuracy")
        precision_scores = pd.Series(metrics["precision_scores"], name="Validation Precision")
        recall_scores = pd.Series(metrics["recall_scores"], name="Validation Recall")

        # plot
        fig, axs = plt.subplots(3, figsize=(5, 15))
        fig.suptitle(f"{model_name}\nPerformance with\nParameter Combination " + str(i + 1))
        x_labels = range(1, len(acc_scores) + 1)

        acc_scores_train.plot(ax=axs[0], c="red", ls=("dashed"))
        precision_scores_train.plot(ax=axs[1], c="blue", ls=("dashed"))
        recall_scores_train.plot(ax=axs[2], c="green", ls=("dashed"))

        acc_scores.plot(ax=axs[0], c="red")
        precision_scores.plot(ax=axs[1], c="blue")
        recall_scores.plot(ax=axs[2], c="green")

        axs[0].set_title("Accuracy Score")
        axs[0].legend()
        min = np.min([acc_scores_train.min(), acc_scores.min()])
        max = np.max([acc_scores_train.max(), acc_scores.max()])
        axs[0].set_ylim([min, max])
        axs[0].set_xticks(range(len(acc_scores)))
        axs[0].set_xticklabels(x_labels)
        axs[0].set_xlabel("Epochs")

        axs[1].set_title("Precision Score")
        axs[1].legend()
        min = np.min([precision_scores_train.min(), precision_scores.min()])
        max = np.max([precision_scores_train.max(), precision_scores.max()])
        axs[1].set_ylim([min, max])
        axs[1].set_xticks(range(len(acc_scores)))
        axs[1].set_xticklabels(x_labels)
        axs[1].set_xlabel("Epochs")

        axs[2].set_title("Recall Score")
        axs[2].legend()
        min = np.min([recall_scores_train.min(), recall_scores.min()])
        max = np.max([recall_scores_train.max(), recall_scores.max()])
        axs[2].set_ylim([min, max])
        axs[2].set_xticks(range(len(acc_scores)))
        axs[2].set_xticklabels(x_labels)
        axs[2].set_xlabel("Epochs")

        plt.tight_layout(pad=3)
        plt.savefig("visuals/" + model_name + "_combi_" + str(i + 1))
