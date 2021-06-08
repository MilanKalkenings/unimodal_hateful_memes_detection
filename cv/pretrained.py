import tools
import pandas as pd
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW
from torchvision import models


class PretrainedClassifier(nn.Module):
    """
    A binary classifier based on a pretrained component.
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
        super(PretrainedClassifier, self).__init__()
        self.pretrained_component = pretrained_component
        self.linear = nn.Linear(in_features=linear_size, out_features=1)  # binary classification -> 1 out feature
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pretrained_component(x)
        x = self.linear(x)
        return self.sigmoid(x)


class PretrainedWrapper:

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
        device = parameters["device"]
        loader = self.preprocess(data=data, parameters=parameters)["loader"]

        pretrained_component = parameters["pretrained_component"]
        linear_size = parameters["linear_size"]
        model = PretrainedClassifier(pretrained_component=pretrained_component, linear_size=linear_size).to(device)
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
        and the respective values
        :param int verbose: defines the amount of prints made during the call. The higher, the more prints
        :return: The trained model
        """
        # extract the parameters
        n_epochs = best_parameters["n_epochs"]
        lr = best_parameters["lr"]
        device = parameters["device"]
        pretrained_component = parameters["pretrained_component"]
        linear_size = parameters["linear_size"]

        train_loader = self.preprocess(data=train_data, parameters=best_parameters)["loader"]
        model = PretrainedClassifier(pretrained_component=pretrained_component, linear_size=linear_size).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        loss_func = nn.BCELoss()

        # train loop
        for epoch in range(n_epochs):
            print("=== Epoch", epoch + 1, "/", n_epochs, "===")
            model.train()
            for batch in train_loader:
                x_batch, y_batch = batch
                # model(x) = model.__call__(x) performs forward (+ more)
                probas = torch.flatten(model(x=x_batch))
                model.zero_grad()  # reset gradients from last step
                batch_loss = loss_func(probas, y_batch)  # calculate loss
                batch_loss.backward()  # calculate gradients
                optimizer.step()  # update parameters

            if verbose > 0:
                print("Metrics on training data after epoch", epoch + 1, ":")
                self.predict(model=model, data=train_data, parameters=best_parameters)
        return {"model": model}

    def evaluate_hyperparameters(self, folds, parameters, verbose=2):
        """
        Evaluates the given parameters on multiple folds using k-fold cross validation.

        :param list folds: a list of pd.DataFrames. Each of the DataFrames contains one fold of the data available
        during the training time.
        :param dict parameters: a dictionary containing one combination of  parameters.
         The dictionary has at least the keys "n_epochs", "lr", "linear_size",
        "conv_ch1", "conv_ch2", "kernel_size", "pooling_size", "device", "transform_pipe",
        and the respective values
        :param int verbose: defines the amount of prints made during the call. The higher, the more prints
        :return: a dictionary having the keys "acc_scores", "f1_scores" and "parameters", having the accuracy score
        for each fold, the f1 score of each fold and the used parameters as values
        """
        val_acc_scores = []
        val_f1_scores = []

        device = parameters["device"]
        n_epochs = parameters["n_epochs"]
        lr = parameters["lr"]
        pretrained_component = parameters["pretrained_component"]
        linear_size = parameters["linear_size"]

        loss_func = nn.BCELoss()
        for fold_id in range(len(folds)):
            print("=== Fold", fold_id + 1, "/", len(folds), "===")
            sets = tools.train_val_split(data_folds=folds, val_fold_id=fold_id)
            train = sets["train"]
            val = sets["val"]
            preprocessed = self.preprocess(data=train, parameters=parameters)
            train_loader = preprocessed["loader"]

            model = PretrainedClassifier(pretrained_component=pretrained_component, linear_size=linear_size).to(device)
            optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

            for epoch in range(n_epochs):
                model.train()
                print("=== Epoch", epoch + 1, "/", n_epochs, "===")
                for i, batch in enumerate(train_loader):
                    x_batch, y_batch = batch
                    probas = torch.flatten(model(x=x_batch))  # forward
                    model.zero_grad()
                    batch_loss = loss_func(probas, y_batch)  # calculate loss
                    batch_loss.backward()  # calculate gradients
                    optimizer.step()  # update parameters
                    if verbose > 1:
                        if i % int(len(train_loader) / len(train_loader)) == 0:
                            print("iteration", i + 1, "/", len(train_loader), "; loss:", batch_loss.item())
                if verbose > 0:
                    print("Metrics on training data after epoch", epoch + 1, ":")
                    self.predict(model=model, data=val, parameters=parameters)

            # validate performance of this fold-split after all epochs are performed:
            print("Metrics using fold", fold_id + 1, "as validation fold:")
            metrics = self.predict(model=model, data=val, parameters=parameters)
            val_acc_scores.append(metrics["acc"])
            val_f1_scores.append(metrics["f1"])
        return {"acc_scores": val_acc_scores, "f1_scores": val_f1_scores, "parameters": parameters}

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
        loader = self.preprocess(data=data, parameters=parameters)["loader"]
        for batch in loader:
            x_batch, y_batch = batch
            with torch.no_grad():
                probas = torch.flatten(model(x=x_batch))
            metrics = tools.evaluate(y_true=y_batch, y_probas=probas)
            acc += metrics["acc"]
            f1 += metrics["f1"]
        acc /= len(loader)
        f1 /= len(loader)

        print("Accuracy:", acc)
        print("F1-Score:", f1)
        return {"acc": acc, "f1": f1}


device = tools.select_device()
print("device:", device)

folds = tools.read_folds(prefix="undersampled_img", read_path="../../data/folds_cv", test_fold_id=0)
train_folds = folds["available_for_train"]
test_fold = folds["test"]

transform_pipe = transforms.Compose([transforms.RandomCrop(size=[512, 512], pad_if_needed=True), transforms.ToTensor()])
parameters = {"transform_pipe": transform_pipe,
              "pretrained_component": models.mobilenet_v3_large(pretrained=True),  # a pretrained model
              "linear_size": 1_000,  # has to match the output size of the pretrained model!
              "n_epochs": 3,
              "lr": 0.0001,
              "batch_size": 16,
              "device": device}

train_data = train_folds[0]
for i in range(1, len(train_folds) - 1):
    pd.concat([train_data, train_folds[i]], axis=0)

pretrained_wrapper = PretrainedWrapper()
# pretrained_wrapper.evaluate_hyperparameters(folds=train_folds, parameters=parameters)

best_cnn = pretrained_wrapper.fit(train_data=train_data, best_parameters=parameters)["model"]
print("PERFORMANCE ON TEST")
pretrained_wrapper.predict(model=best_cnn, data=test_fold, parameters=parameters)
