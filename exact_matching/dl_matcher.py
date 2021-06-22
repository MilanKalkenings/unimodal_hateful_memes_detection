import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models
from torchvision import transforms
from transformers import AdamW

import tools


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
        Creates a DataLoader given the data and some parameters.

        :param pd.DataFrame data: a DataFrame containing the paths to image files and the labels of the
        respective image.
        :param dict parameters: a dictionary containing the parameters defined in tools.parameters_pretrained
        :return: A DataLoader That loads images transformed by the transformation pipeline and the respective targets.
        """
        transform_pipe = parameters["transform_pipe"]
        batch_size = parameters["batch_size"]
        device = parameters["device"]
        custom_dataset = tools.CustomDataset(data=data, transform_pipe=transform_pipe, device=device)
        sampler = RandomSampler(data_source=custom_dataset)
        loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, sampler=sampler)
        return {"loader": loader}

    def fit(self, train_data, best_parameters):
        """
        Trains a PretrainedClassifier on train_data using a set of parameters.

        :param pd.DataFrame train_data: data on which the model has to be trained
        :param dict best_parameters: a dictionary containing the parameters defined in tools.parameters_pretrained
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

        train_data, val_data = train_test_split(train_data, test_size=0.2)
        train_data.index = range(len(train_data))
        val_data.index = range(len(val_data))

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
                optimizer.zero_grad()
                probas = torch.flatten(model(x=x_batch))
                batch_loss = loss_func(probas, y_batch)  # calculate loss
                batch_loss.backward()  # calculate gradients
                optimizer.step()

            print("Metrics on training data after epoch", epoch, ":")
            self.predict(model=model, data=train_data, parameters=best_parameters)
            print("Metrics on validation data after epoch", epoch, ":")
            self.predict(model=model, data=val_data, parameters=best_parameters)
            print("\n")
        return {"model": model}

    def predict(self, model, data, parameters):
        """
        Predicts the labels of a dataset and evaluates the results against the ground truth.

        :param CNNClassifier model: a trained CNNClassifier
        :param pd.DataFrame data: a dataset on which the prediction has to be performed
        :param dict parameters: a dictionary containing the parameters defined in tools.parameters_pretrained
        :return: a dictionary containing the accuracy, prediction,
        and recall score of the models predictions on the data
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

    # TODO abandoned
    def compare_representations(self, data, detected, model, parameters):
        """
        Creates the representations given by the pretrained component for the given data.

        :param pd.DataFrame data: contains the data that has to be classified
        :param pd.DataFrame detected: contains the already detected hateful memes
        :param model: an instance of ExactClassifier
        :param dict parameters: the parameters defined by tools.parameters_exact_wrapper
        """
        # representation of >data<:
        parameters["batch_size"] = 1
        loader_data = self.preprocess(data=data, parameters=parameters)["loader"]
        representations_data = []
        for x, y in loader_data:
            representations_data.append(model.pretrained_representation(x))
        reps_data = pd.Series(representations_data)

        # representation of all detected hateful memes:
        loader_detected = self.preprocess(data=detected, parameters=parameters)
        representations_detected = []
        for x, y in loader_detected:
            representations_detected.append(model.pretrained_representation(x))
        reps_detected = pd.Series(representations_detected)

        # compute similarity of >datapoint< to all detected hateful memes
        cos = nn.CosineSimilarity()

        def cos_sim(x, y, cos):
            """
            Calculates the cosine similarity of two tensors.

            :param torch.Tensor x: first tensor
            :param torch.Tensor y: second tensor
            :param nn.CosineSimilarity cos: a module to calculate the cosine similarity of two tensors
            :return: the cosine similarity of the two tensors
            """
            return cos(x.flatten(), y.flatten())

        def reps_to_preds(reps_data, reps_detected, thresh):
            """
            Predicts whether a meme is already known to be hateful or not, and evaluates the result using
            accuracy score, prediction score and recall score.

            :param pd.Series reps_data: representations of memes that have to be classified
            :param pd.Series reps_detected: representations of memes that are known to be hateful
            :param float thresh: the threshold determines how similar two representations have to be in order to
            decide that they belong to the same meme
            """
            preds = []
            for i in range(len(reps_data)):
                pred = 0
                for j in range(len(reps_detected)):
                    if cos_sim(x=reps_data[i], y=reps_detected[j], cos=cos) > thresh:
                        pred = 1
                        break
                preds.append(pred)
            return np.array(preds)

        for thresh in np.arange(start=0, stop=1, step=0.1):
            print(thresh)
            preds = reps_to_preds(reps_data=reps_data, reps_detected=reps_detected, thresh=thresh)
            print("Accuracy:", accuracy_score(y_true=data["label"], y_pred=preds))
            print("Precision:", precision_score(y_true=data["label"], y_pred=preds))
            print("Recall:", recall_score(y_true=data["label"], y_pred=preds))


# read the datasets
data = tools.read_data(detected_share=0.05)
train_data = data["train"]
val_data = data["val"]
test_data = data["test"]
detected = data["detected"]
non_detected = data["non_detected"]

# define the parameters
device = tools.select_device()
print("device:", device)
color_jitter = transforms.ColorJitter(brightness=[0, 2], hue=[-0.1, 0.1], contrast=[0, 2], saturation=[0, 2])
random_crop = transforms.RandomCrop(size=[256, 256], pad_if_needed=True)
transform_pipe = transforms.Compose([random_crop, color_jitter, transforms.ToTensor()])
parameters = tools.parameters_exact_wrapper(n_epochs=100,
                                            lr=0.0001,
                                            batch_size=32,
                                            transform_pipe=transform_pipe,
                                            pretrained_component=models.mobilenet_v3_large(pretrained=True),
                                            linear_size=8,
                                            freeze_epochs=[],
                                            unfreeze_epochs=[],
                                            device=device)

# use the model
exact_wrapper = ExactWrapper()

best_exact = exact_wrapper.fit(train_data=train_data, best_parameters=parameters)["model"]
torch.save(best_exact, "best_exact.pt")
print("\nPERFORMANCE ON TEST")
exact_wrapper.predict(model=best_exact, data=test_data, parameters=parameters)


best_exact = torch.load("best_exact.pt")

# using the embeddings
exact_wrapper.compare_representations(data=test_data, detected=detected, model=best_exact, parameters=parameters)
