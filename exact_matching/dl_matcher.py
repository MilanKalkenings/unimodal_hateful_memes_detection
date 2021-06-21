import numpy as np

import tools
import pandas as pd
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from torchvision import models


class ExactClassifier(nn.Module):
    """
    A binary classifier based on a pretrained component used for exact matching.
    A variety of pretrained models can be used.
    Some of which are (tested):
    resnet101,
    resnet18,
    resnext101_32x8d,
    googlelenet,
    alexnet,
    mobilenet_v3_large
    """

    def __init__(self, linear_size, pretrained_component):
        """
        Constructor.

        :param int linear_size: size of the second linear layer
        :param pretrained_component: a pretrained model for image classification. All pretrained models provided by
        PyTorch provide an output tensor of size 1_000.
        """
        super(ExactClassifier, self).__init__()
        self.pretrained_component = pretrained_component
        self.linear1 = nn.Linear(in_features=1_000, out_features=linear_size)
        self.linear2 = nn.Linear(in_features=linear_size, out_features=1)  # binary classification -> 1 out feature
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pretrained_component(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return self.sigmoid(x)

    def freeze_pretrained(self):
        """
        Freezes all parameters of the pretrained component.
        Frozen parameters are not trained, because no gradient is created with respect to them.
        """
        for param in self.pretrained_component.parameters():
            param.requires_grad = False

    def unfreeze_pretrained(self):
        """
        Unfreezes all parameters of the pretrained component.
        Parameters are unfrozen by default and can be frozen by the function >freeze_pretrained<
        """
        for param in self.pretrained_component.parameters():
            param.requires_grad = False

    def pretrained_representation(self, x):
        """
        Performs the forward poss of the pretrained component only.

        :param tensor x: an imput
        :return: The representation of the input created by the the pretrained component.
        """
        return self.pretrained_component(x)


class ExactWrapper:

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
        sampler = parameters["sampler"]
        custom_dataset = tools.CustomDataset(data=data, transform_pipe=transform_pipe, device=device)
        sampler = sampler(data_source=custom_dataset)
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
        device = parameters["device"]
        loader = self.preprocess(data=data, parameters=parameters)["loader"]

        pretrained_component = parameters["pretrained_component"]
        linear_size = parameters["linear_size"]
        model = ExactClassifier(pretrained_component=pretrained_component, linear_size=linear_size).to(device)
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
        "freeze_epochs", "unfreeze_epochs", and the respective values.
        :param int verbose: defines the amount of prints made during the call. The higher, the more prints
        :return: The trained model
        """
        # extract the parameters
        n_epochs = best_parameters["n_epochs"]
        lr = best_parameters["lr"]
        device = best_parameters["device"]
        pretrained_component = best_parameters["pretrained_component"]
        linear_size = best_parameters["linear_size"]
        freeze_epochs = best_parameters["freeze_epochs"]
        unfreeze_epochs = best_parameters["unfreeze_epochs"]
        accumulation = best_parameters["accumulation"]

        train_loader = self.preprocess(data=train_data, parameters=best_parameters)["loader"]
        model = ExactClassifier(pretrained_component=pretrained_component, linear_size=linear_size).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        loss_func = nn.BCELoss()

        # train loop
        for epoch in range(1, n_epochs + 1):
            print("=== Epoch", epoch, "/", n_epochs, "===")
            if epoch in freeze_epochs:
                print("Freeze")
                model.freeze_pretrained()
            elif epoch in unfreeze_epochs:
                print("Unfreeze")
                model.unfreeze_pretrained()

            model.train()
            for i, batch in enumerate(train_loader):
                x_batch, y_batch = batch
                probas = torch.flatten(model(x=x_batch))
                batch_loss = loss_func(probas, y_batch)  # calculate loss
                batch_loss /= accumulation
                batch_loss.backward()  # calculate gradients

                if ((i + 1) % accumulation == 0):
                    optimizer.step()
                    optimizer.zero_grad()


            if verbose > 0:
                print("Metrics on training data after epoch", epoch, ":")
                self.predict(model=model, data=train_data, parameters=best_parameters)
        return {"model": model}

    def predict(self, model, data, parameters):
        """
        Predicts the labels of a dataset and evaluates the results against the ground truth.

        :param CNNClassifier model: a trained CNNClassifier
        :param pd.DataFrame data: a dataset on which the prediction has to be performed
        :param dict parameters: a dictionary having at least the keys "max_seq_len", "batch_size", "x_name", "y_name",
        "device", and the respective values.
        :return: a dictionary containing the f1_score and the accuracy_score of the models predictions on the data
        """
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

    def pretrained_representations(self, data, model, parameters):
        """
        Creates the representations given by the pretrained component for the given data.

        :param pd.DataFrame data: contains the data for which the representations have to be calculated.
        :param dict parameters: #TODO
        :return: A pd.Series containing the returned representations of the input.
        """
        parameters["batch_size"] = 1
        parameters["sampler"] = SequentialSampler
        loader = self.preprocess(data=data, parameters=parameters)["loader"]

        representations = []
        for x, y in loader:
            representations.append(model.pretrained_representation(x))
        return pd.Series(representations, name="representations")


    def representation_similarity(self, representation1, representation2):
        """
        Calculates the

        :param torch.Tensor representation1:
        :param torch.Tensor representation2:
        :return:
        """
        cos = nn.CosineSimilarity()
        return cos(representation1.flatten(), representation2.flatten())


# read the datasets
data = tools.read_data(detected_share=0.05)
train_data = data["train"]
val_data = data["val"]
test_data = data["test"]
detected = data["detected"]
non_detected = data["non_detected"]
print("Train Data Distribution of the target 'detected':")
print(train_data["detected"].value_counts())

# define the parameters
device = tools.select_device()
print("device:", device)
color_jitter = transforms.ColorJitter(brightness=[0, 2], hue=[-0.1, 0.1], contrast=[0, 2], saturation=[0, 2])
random_crop = transforms.RandomCrop(size=[256, 256], pad_if_needed=True)
transform_pipe = transforms.Compose([random_crop, color_jitter, transforms.ToTensor()])
parameters = {"transform_pipe": transform_pipe,
              "pretrained_component": models.mobilenet_v3_large(pretrained=True),  # a pretrained model
              "linear_size": 1024,
              "n_epochs": 50,
              "lr": 0.001,
              "batch_size": 32,
              "device": device,
              "freeze_epochs": [],
              "unfreeze_epochs": [],
              "sampler": RandomSampler,
              "accumulation": 1}


# use the model
exact_wrapper = ExactWrapper()
best_exact = exact_wrapper.fit(train_data=train_data, best_parameters=parameters)["model"]
print("\nPERFORMANCE ON TEST")
exact_wrapper.predict(model=best_exact, data=test_data, parameters=parameters)
