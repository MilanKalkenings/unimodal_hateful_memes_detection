import tools
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset
from transformers import AdamW


device = tools.select_device()
print("device:", device)

class CNNClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(in_features=16*6*3, out_features=16*3)
        self.linear2 = nn.Linear(in_features=16*3, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.linear1(x)
        return self.linear2(x)


class CustomDataset(Dataset):
    def __init__(self, data, transform_pipe, x_name="img", y_name="label", device="cuda"):
        self.data = data
        self.transform_pipe = transform_pipe
        self.x_name = x_name
        self.y_name = y_name
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_path = "../../data/hateful_memes_data/" + self.data[self.x_name].iloc[i]
        x = self.transform_pipe(Image.open(img_path)).to(self.device)
        y = torch.tensor(self.data[self.y_name]).to(self.device)
        return [x, y]


class CNNWrapper:

    @staticmethod
    def preprocess(fold_id, transform_pipe, batch_size=32, device="cuda"):
        fold_path = "../../data/folds_cv/fold" + str(fold_id)
        dataset = datasets.ImageFolder(fold_path, transform=transform_pipe)
        sampler = RandomSampler(data_source=dataset)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        return {"loader": loader}

    def fit(self, fold_id, best_parameters, transform_pipe, verbose=0):
        # extract the parameters
        n_epochs = best_parameters["n_epochs"]
        lr = best_parameters["lr"]
        batch_size = best_parameters["batch_size"]
        device = best_parameters["device"]

        train_loader = self.preprocess(fold_id=fold_id,
                                       transform_pipe=transform_pipe,
                                       batch_size=batch_size,
                                       device=device)["loader"]

        model = CNNClassifier().to(device)
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        loss_func = nn.BCELoss()

        # train loop
        for epoch in range(n_epochs):
            print("=== Epoch", epoch + 1, "/", n_epochs, "===")
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
                self.predict(model=model, fold_id=fold_id, transform_pipe=transform_pipe, parameters=best_parameters)
        return model

    def predict(self, model, fold_id, transform_pipe, parameters):
        """
        Predicts the labels of a dataset and evaluates the results against the ground truth.

        :param BertClassifier model: a trained BERTClassifier
        :param pd.DataFrame data: a dataset on which the prediction has to be performed
        :param dict parameters: a dictionary having at least the keys "max_seq_len", "batch_size", "x_name", "y_name",
        "device", and the respective values.
        :return: a dictionary containing the f1_score and the accuracy_score of the models predictions on the data
        """
        # extract the parameters
        batch_size = parameters["batch_size"]
        device = parameters["device"]

        acc = 0
        f1 = 0
        loader = self.preprocess(fold_id=fold_id,
                                 transform_pipe=transform_pipe,
                                 batch_size=batch_size,
                                 device=device)["loader"]
        for batch in loader:
            x_batch, y_batch= batch
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

#cnn_wrapper = CNNWrapper()
transform_pipe = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(128), transforms.ToTensor()])
#loader = cnn_wrapper.preprocess(fold_id=0, transform_pipe=transform_pipe)
#print(next(iter(loader["loader"]))[0].size())



#parameters = {"n_epochs": 10, "batch_size": 32, "lr": 0.001, "device": device}
#cnn_wrapper.fit(fold_id=0, transform_pipe=transform_pipe, best_parameters=parameters)

data = tools.read_folds(prefix="img", read_path="../../data/folds_cv")["test"]
#print(data)
custom_dataset = CustomDataset(data=data, transform_pipe=transform_pipe, device=device)
sampler = RandomSampler(data_source=custom_dataset)
loader = DataLoader(dataset=custom_dataset, sampler=sampler)
print(next(iter(loader)))
