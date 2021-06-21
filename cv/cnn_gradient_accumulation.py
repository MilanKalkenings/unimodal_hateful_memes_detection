import tools
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW
from sklearn.metrics import accuracy_score

device = tools.select_device()
print("device:", device)


class CNNClassifier(nn.Module):
    """
    A CNN-based classifier that is capable of performing dropout-regularization.
    """

    def __init__(self, conv_ch1, conv_ch2, linear_size, kernel_size, pooling_size, linear_input_size=None):
        """
        Constructor.

        :param int conv_ch1: number of output channels of the first convolutional layer
        :param int conv_ch2: number of output channels of the second convolutional layer
        :param int linear_size: number of outputs the second linear layer expects from the first linear layer
        :param int kernel_size: width and height of each convolutional kernel in the model
        :param int pooling_size: size of the pooling window
        :param int linear_input_size: number of outputs the first linear layer expects from the convolutional
        part.
        """
        super(CNNClassifier, self).__init__()
        self.linear_input_size = linear_input_size
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pooling_size)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv_ch1, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=conv_ch1, out_channels=conv_ch2, kernel_size=kernel_size)

        if linear_input_size:  # evaluates as False if linear_input_size is None
            self.dropout = nn.Dropout(p=0.3)  # probability of an input being ignored
            self.linear1 = nn.Linear(in_features=linear_input_size, out_features=linear_size)
            self.linear2 = nn.Linear(in_features=linear_size, out_features=1)
            self.sigmoid = nn.Sigmoid()

    def conv_part(self, x):
        """
        Calculates the convolutional part of the forward pass.

        :param torch.Tensor x: input data
        :return: input representations after the convolutional part.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

    def scalars_after_conv(self, x):
        """
        Calculates how many scalars, i.e. tensors of rank 0 are in the output of the convolutional component.

        :param torch.Tensor x: one batch of input/observations
        :return: the number of rank 0 tensors in the output of the convolutional component
        """
        x = self.conv_part(x)
        size = x.size()  # batch_size x channel_size x width x height
        n_channels = size[1]
        width = size[2]
        height = size[3]
        return n_channels * width * height

    def forward(self, x):
        """
        performs the forward pass.

        :param  torch.Tensor x: the input/observations per batch
        :return: the prediction of the whole batch
        """
        x = self.conv_part(x)

        if self.linear_input_size:
            x = x.view(-1, self.linear_input_size)  # flatten out for the linear layers
            x = self.dropout(x)
            x = self.linear1(x)
            x = self.linear2(x)
        return self.sigmoid(x)


class CNNWrapper:

    @staticmethod
    def preprocess(data, parameters):
        """
        Creates a DataLoader given teh data and some parameters.

        :param pd.DataFrame data: a DataFrame containing the paths to image files and the labels of the
        respective image.
        :param dict parameters: a dictionary containing at least the parameters "transform_pipe", "batch_size",
        "device", and the respective values.
        :return: A DataLoader That loads images transformed by the transformation pipeline and the respective targets.
        """
        transform_pipe = parameters["transform_pipe"]
        batch_size = parameters["batch_size"]
        device = parameters["device"]
        custom_dataset = tools.CustomDataset(data=data, transform_pipe=transform_pipe, device=device)
        sampler = RandomSampler(data_source=custom_dataset)
        loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, sampler=sampler)
        return {"loader": loader}

    def find_linear_input_size(self, data, parameters):
        """
        Finds the number of parameters the first hidden layer that is attached to the convolutional
        component of the model has to have given some data.

        :param pd.DataFrame data: data on which the model has to be trained
        :param dict parameters: a dictionary containing the best parameters
        (found using evaluate hyperparameters of this class). The dictionary has at least the keys ,
        "linear_size", "conv_ch1", "conv_ch2", "kernel_size", "pooling_size", "device", "transform_pipe",
        and the respective values
        :return: The number of neurons that the first linear layer of the model needs to have
        """
        linear_size = parameters["linear_size"]
        conv_ch1 = parameters["conv_ch1"]
        conv_ch2 = parameters["conv_ch2"]
        kernel_size = parameters["kernel_size"]
        pooling_size = parameters["pooling_size"]
        device = parameters["device"]

        loader = self.preprocess(data=data, parameters=parameters)["loader"]

        model = CNNClassifier(linear_size=linear_size,
                              conv_ch1=conv_ch1,
                              conv_ch2=conv_ch2,
                              kernel_size=kernel_size,
                              pooling_size=pooling_size).to(device)
        example_batch = next(iter(loader))
        example_x = example_batch[0]
        return model.scalars_after_conv(x=example_x)

    def fit(self, train_data, best_parameters, verbose=2):
        """
        Trains an CNNClassifier on train_data using a set of parameters.

        :param pd.DataFrame train_data: data on which the model has to be trained
        :param dict best_parameters: a dictionary containing the best parameters
        (found using evaluate hyperparameters of this class). The dictionary has at least the keys "n_epochs", "lr",
        "linear_size", "conv_ch1", "conv_ch2", "kernel_size", "pooling_size", "device", "transform_pipe",
        "accumulation", and the respective values
        :param int verbose: defines the amount of prints made during the call. The higher, the more prints
        :return: The trained model
        """
        # extract the parameters
        accumulation = best_parameters["accumulation"]
        linear_size = best_parameters["linear_size"]
        conv_ch1 = best_parameters["conv_ch1"]
        conv_ch2 = best_parameters["conv_ch2"]
        kernel_size = best_parameters["kernel_size"]
        pooling_size = best_parameters["pooling_size"]
        n_epochs = best_parameters["n_epochs"]
        lr = best_parameters["lr"]
        device = best_parameters["device"]

        train_loader = self.preprocess(data=train_data, parameters=best_parameters)["loader"]

        linear_input_size = self.find_linear_input_size(data=train_data, parameters=best_parameters)
        model = CNNClassifier(linear_size=linear_size,
                              conv_ch1=conv_ch1,
                              conv_ch2=conv_ch2,
                              linear_input_size=linear_input_size,
                              kernel_size=kernel_size,
                              pooling_size=pooling_size).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        loss_func = nn.BCELoss()

        # train loop
        for epoch in range(n_epochs):
            print("=== Epoch", epoch + 1, "/", n_epochs, "===")
            model.train()
            for i, batch in enumerate(train_loader):
                x_batch, y_batch = batch
                # model(x) = model.__call__(x) performs forward (+ more)
                probas = torch.flatten(model(x=x_batch))
                batch_loss = loss_func(probas, y_batch)  # calculate loss
                batch_loss /= accumulation
                batch_loss.backward()  # calculate gradients

                if ((i + 1) % accumulation == 0):
                    optimizer.step()  # update parameters
                    optimizer.zero_grad()  # clear the gradient

            if verbose > 0:
                print("Metrics on training data after epoch", epoch + 1, ":")
                self.predict(model=model, data=train_data, parameters=best_parameters)
        return {"model": model}

    def demo_one_batch(self, train_data, best_parameters):
        """
        Trains an CNNClassifier on one batch of the train_data using a set of parameters.
        This function is used to demonstrate that the model can learn the
        patterns of some given data. It will vastly overfit the batch, if the model works properly.
        Functions like these are helpful for debugging neural networks.

        :param pd.DataFrame train_data: data on which the model has to be trained
        :param dict best_parameters: a dictionary containing the best parameters
        (found using evaluate hyperparameters of this class). The dictionary has at least the keys "n_epochs", "lr",
        "linear_size", "conv_ch1", "conv_ch2", "kernel_size", "pooling_size", "device", "transform_pipe",
        "accumulation", and the respective values
        """
        # extract the parameters
        linear_size = best_parameters["linear_size"]
        conv_ch1 = best_parameters["conv_ch1"]
        conv_ch2 = best_parameters["conv_ch2"]
        kernel_size = best_parameters["kernel_size"]
        pooling_size = best_parameters["pooling_size"]
        n_epochs = best_parameters["n_epochs"]
        lr = best_parameters["lr"]
        device = best_parameters["device"]
        accumulation = best_parameters["accumulation"]

        train_loader = self.preprocess(data=train_data, parameters=best_parameters)["loader"]
        batch = next(iter(train_loader))

        linear_input_size = self.find_linear_input_size(data=train_data, parameters=best_parameters)
        model = CNNClassifier(linear_size=linear_size,
                              conv_ch1=conv_ch1,
                              conv_ch2=conv_ch2,
                              linear_input_size=linear_input_size,
                              kernel_size=kernel_size,
                              pooling_size=pooling_size).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        loss_func = nn.BCELoss()

        # train loop
        for epoch in range(n_epochs):
            print("=== Epoch", epoch + 1, "/", n_epochs, "===")
            model.train()
            x_batch, y_batch = batch
            # model(x) = model.__call__(x) performs forward (+ more)
            probas = torch.flatten(model(x=x_batch))
            batch_loss = loss_func(probas, y_batch)  # calculate loss
            batch_loss /= accumulation
            batch_loss.backward()  # calculate gradients

            if (epoch + 1) % accumulation == 0:
                optimizer.step()  # update parameters
                optimizer.zero_grad()  # clear the gradient

            preds_batch_np = np.round(probas.cpu().detach().numpy())
            y_batch_np = y_batch.cpu().detach().numpy()
            print("accuracy:", accuracy_score(y_true=y_batch_np, y_pred=preds_batch_np))

    def find_max_img_sizes(self, data, parameters):
        """
        Finds the maximum width and height of all the images represented by the paths in data.
        This function can be used to determine the "size" parameter in transformation pipelines.

        :param pd.DataFrame data: a dataframe representing an image dataset having at least one column with image paths
        and one column with classification labels.
        :param dict parameters: a dictionary having all necessary model parameters of the wrapped model class
        as keys and the respective values.
        :return: maximum width and maximum height over all images in data
        """
        parameters_shadow = parameters.copy()
        transform_pipe = transforms.Compose([transforms.ToTensor()])
        parameters_shadow["transform_pipe"] = transform_pipe
        parameters_shadow["batch_size"] = 1
        loader = self.preprocess(data=data, parameters=parameters_shadow)["loader"]
        max_width = 0
        max_height = 0
        for batch in loader:
            width = batch[0].size(2)
            height = batch[0].size(3)
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
        return {"height": max_height, "width": max_width}

    def evaluate_hyperparameters(self, folds, parameters):
        """
        Evaluates the given parameters on multiple folds using k-fold cross validation.

        :param list folds: a list of pd.DataFrames. Each of the DataFrames contains one fold of the data available
        during the training time.
        :param dict parameters: a dictionary containing one combination of  parameters.
         The dictionary has at least the keys "n_epochs", "lr", "linear_size",
        "conv_ch1", "conv_ch2", "kernel_size", "pooling_size", "device", "transform_pipe", "freeze_epochs",
        "unfreeze_epochs", and the respective values.
        :return: a dictionary having the keys "acc_scores", "f1_scores" and "parameters", having the accuracy score
        and the f1 score after each epoch averaged over all folds, and the used parameters as values.
        """
        print("\nEvaluating Hyperparameters:\n", parameters)
        n_epochs = parameters["n_epochs"]
        lr = parameters["lr"]
        linear_size = parameters["linear_size"]
        conv_ch1 = parameters["conv_ch1"]
        conv_ch2 = parameters["conv_ch2"]
        kernel_size = parameters["kernel_size"]
        pooling_size = parameters["pooling_size"]
        device = parameters["device"]
        accumulation = parameters["accumulation"]

        acc_scores = np.zeros(n_epochs)
        f1_scores = np.zeros(n_epochs)
        loss_func = nn.BCELoss()
        for fold_id in range(len(folds)):
            print("=== Fold", fold_id + 1, "/", len(folds), "===")
            sets = tools.train_val_split(data_folds=folds, val_fold_id=fold_id)
            train = sets["train"]
            val = sets["val"]
            preprocessed = self.preprocess(data=train, parameters=parameters)
            train_loader = preprocessed["loader"]
            linear_input_size = self.find_linear_input_size(data=folds[0], parameters=parameters)
            model = CNNClassifier(linear_input_size=linear_input_size,
                                  linear_size=linear_size,
                                  conv_ch1=conv_ch1,
                                  conv_ch2=conv_ch2,
                                  kernel_size=kernel_size,
                                  pooling_size=pooling_size).to(device)  # isolated model per fold
            optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)  # depends on model

            for epoch in range(1, n_epochs + 1):
                model.train()
                print("=== Epoch", epoch, "/", n_epochs, "===")
                for i, batch in enumerate(train_loader):
                    x_batch, y_batch = batch
                    probas = torch.flatten(model(x=x_batch))  # forward
                    batch_loss = loss_func(probas, y_batch)  # calculate loss
                    batch_loss /= accumulation
                    batch_loss.backward()  # calculate gradients

                    if (i + 1) % accumulation == 0:
                        optimizer.step()  # update parameters
                        optimizer.zero_grad()  # clear the gradient

                print("Metrics on training data after epoch", epoch, ":")
                self.predict(model=model, data=train, parameters=parameters)
                print("Metrics on validation data after epoch", epoch, ":")
                metrics = self.predict(model=model, data=val, parameters=parameters)
                acc_scores[epoch - 1] += metrics["acc"]
                f1_scores[epoch - 1] += metrics["f1"]
                print("\n")

        for i in range(n_epochs):
            acc_scores[i] /= len(folds)
            f1_scores[i] /= len(folds)
        return {"acc_scores": acc_scores, "f1_scores": f1_scores, "parameters": parameters}

    def predict(self, model, data, parameters):
        """
        Predicts the labels of a dataset and evaluates the results against the ground truth.

        :param CNNClassifier model: a trained CNNClassifier
        :param pd.DataFrame data: a dataset on which the prediction has to be performed
        :param dict parameters: a dictionary having at least the keys "max_seq_len", "batch_size", "x_name", "y_name",
        "device", and the respective values.
        :return: a dictionary containing the f1_score and the accuracy_score of the models predictions on the data
        """
        # extract the parameters
        model.eval()
        acc = 0
        f1 = 0
        precision = 0
        recall = 0
        loader = self.preprocess(data=data, parameters=parameters)["loader"]
        for batch in loader:
            x_batch, y_batch = batch
            with torch.no_grad():
                probas = torch.flatten(model(x=x_batch))
            metrics = tools.evaluate(y_true=y_batch, y_probas=probas)
            acc += metrics["acc"]
            f1 += metrics["f1"]
            precision += metrics["precision"]
            recall += metrics["recall"]
        acc /= len(loader)
        f1 /= len(loader)
        precision /= len(loader)
        recall /= len(loader)

        print("Accuracy:", acc)
        print("F1-Score:", f1)
        print("Precision:", precision)
        print("Recall:", recall)
        return {"acc": acc, "f1": f1}


# read the datasets
folds = tools.read_folds(prefix="undersampled_img",
                         read_path="../../data/folds_cv",
                         test_fold_id=0)
train_folds = folds["train"]
test_fold = folds["test"]
train_data = train_folds[0]
for i in range(1, len(train_folds) - 1):
    pd.concat([train_data, train_folds[i]], axis=0)

# define the parameters
transform_pipe = transforms.Compose([transforms.RandomCrop(size=[512, 512], pad_if_needed=True),
                                     transforms.ToTensor()])
parameters = {"transform_pipe": transform_pipe,
              "n_epochs": 3,
              "lr": 0.0001,
              "batch_size": 8,
              "device": device,
              "conv_ch1": 8,
              "conv_ch2": 4,
              "linear_size": 16,
              "kernel_size": 3,
              "pooling_size": 2,
              "accumulation": 4}

# use the model
cnn_wrapper = CNNWrapper()
cnn_wrapper.demo_one_batch(train_data=train_data, best_parameters=parameters)
cnn_wrapper.find_max_img_sizes(data=train_data, parameters=parameters)
print(cnn_wrapper.evaluate_hyperparameters(folds=train_folds, parameters=parameters))
best_cnn = cnn_wrapper.fit(train_data=train_data, best_parameters=parameters)["model"]
print("\nPERFORMANCE ON TEST")
cnn_wrapper.predict(model=best_cnn, data=test_fold, parameters=parameters)


'''
# 67 test accuracy and confusion based metrics above 0.65 on undersampled_img
transform_pipe = transforms.Compose([transforms.RandomCrop(size=[512, 512], pad_if_needed=True),
                                     transforms.ToTensor()])
parameters = {"transform_pipe": transform_pipe,
              "n_epochs": 20,
              "lr": 0.0001,
              "batch_size": 128,
              "device": device,
              "conv_ch1": 4,
              "conv_ch2": 2,
              "linear_size": 32,
              "kernel_size": 3,
              "pooling_size": 2}
'''




