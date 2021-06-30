import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW
from transformers import BertModel
from transformers import BertTokenizer

import tools


class BertClassifier(nn.Module):
    """
    BERT-based model for binary sequence classification.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # returns powerful representations of the texts
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(768, 1)  # input of the first custom layer has to match dim of the BERT-output
        self.sigmoid = nn.Sigmoid()  # activation function applied to obtain probabilities

    def forward(self, x, attention_mask):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        bert_output = self.bert(input_ids=x, attention_mask=attention_mask)
        x = self.dropout(bert_output[1])
        x = self.linear(x)
        return self.sigmoid(x)


class BertWrapper:

    @staticmethod
    def preprocess(data, parameters):
        """
        Preprocesses the data of a fold and returns the DataLoaders for the wrapped neural network.

        :param pd.DataFrame data: one fold of data to be processed. Contains a column <x_name> containing text
        sequences and another column <y_name> containing the class labels of the sequence
        :param dict parameters: a dictionary containing the parameters defined in tools.parameters_bert_based
        :return: a dictionary having the key "loader" and the constructed DataLoader as value.
        """
        max_seq_len = parameters["max_seq_len"]
        batch_size = parameters["batch_size"]
        x_name = parameters["x_name"]
        y_name = parameters["y_name"]
        text_col = data[x_name]
        target_col = data[y_name]

        # tokenization:
        tokens = []
        attention_masks = []
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        for text in text_col.values:
            encoded = tokenizer.encode_plus(text,
                                            add_special_tokens=True,  # special tokens for BERT
                                            max_length=max_seq_len,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt",  # return pytorch tensors
                                            return_attention_mask=True)
            tokens.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
        x = torch.cat(tokens, dim=0)  # concat into one big tensor
        x = x.long().to(device)
        attention_masks = torch.cat(attention_masks, dim=0).to(device)
        y = torch.tensor(target_col, dtype=torch.float32).to(device)

        # create loader:
        dataset = TensorDataset(x, y, attention_masks)
        sampler = RandomSampler(dataset)
        return {"loader": DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)}

    def evaluate_hyperparameters(self, folds, parameters):
        """
        Evaluates the given parameters on multiple folds using k-fold cross validation.

        :param list folds: a list of pd.DataFrames. Each of the DataFrames contains one fold of the data available
        during the training time.
        :param dict parameters: a dictionary containing the parameters defined in tools.parameters_bert_based
        :return: a dictionary containing the accuracy and roc-auc scores on both training and validation data
        """
        n_epochs = parameters["n_epochs"]
        lr = parameters["lr"]
        device = parameters["device"]

        acc_scores_train = np.zeros(n_epochs)
        roc_auc_scores_train = np.zeros(n_epochs)

        acc_scores = np.zeros(n_epochs)
        roc_auc_scores = np.zeros(n_epochs)

        loss_func = nn.BCELoss()
        for fold_id in range(len(folds)):
            print("=== Fold", fold_id + 1, "/", len(folds), "===")
            sets = tools.train_val_split(data_folds=folds, val_fold_id=fold_id)
            train = sets["train"]
            val = sets["val"]
            train_loader = self.preprocess(data=train, parameters=parameters)["loader"]
            model = BertClassifier().to(device)  # create one model per fold split (isolated training)
            optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)  # depends on model

            for epoch in range(1, n_epochs + 1):
                print("=== Epoch", epoch, "/", n_epochs, "===")
                model.train()
                for batch in train_loader:
                    x_batch, y_batch, attention_mask = batch
                    probas = torch.flatten(model(x=x_batch, attention_mask=attention_mask))  # forward
                    model.zero_grad()
                    batch_loss = loss_func(probas, y_batch)  # calculate loss
                    batch_loss.backward()  # calculate gradients
                    optimizer.step()  # update parameters

                print("Metrics on training data after epoch", epoch, ":")
                metrics = self.predict(model=model, data=train, parameters=parameters)
                acc_scores_train[epoch - 1] += metrics["acc"]
                roc_auc_scores_train[epoch - 1] += metrics["roc_auc"]

                print("Metrics on validation data after epoch", epoch, ":")
                metrics = self.predict(model=model, data=val, parameters=parameters)
                acc_scores[epoch - 1] += metrics["acc"]
                roc_auc_scores[epoch - 1] += metrics["roc_auc"]
                print("\n")

        for i in range(n_epochs):
            acc_scores_train[i] /= len(folds)
            roc_auc_scores_train[i] /= len(folds)

            acc_scores[i] /= len(folds)
            roc_auc_scores[i] /= len(folds)

        return {"acc_scores_train": acc_scores_train, "acc_scores": acc_scores,
                "roc_auc_scores_train": roc_auc_scores_train, "roc_auc_scores": roc_auc_scores}

    def predict(self, model, data, parameters):
        """
        Predicts the labels of a dataset and evaluates the results against the ground truth.

        :param BertClassifier model: a trained BERTClassifier
        :param pd.DataFrame data: a dataset on which the prediction has to be performed
        :param dict parameters: a dictionary containing the parameters defined in tools.parameters_bert_based
        :return: a dictionary containing the accuracy and roc-auc score of the models predictions on the data
        """
        model.eval()
        acc = 0
        roc_auc = 0
        loader = self.preprocess(data=data, parameters=parameters)["loader"]
        for batch in loader:
            x_batch, y_batch, attention_mask = batch
            with torch.no_grad():
                probas = torch.flatten(model(x=x_batch, attention_mask=attention_mask))
            metrics = tools.evaluate(y_true=y_batch, y_probas=probas)
            acc += metrics["acc"]
            roc_auc += metrics["roc_auc"]

        acc /= len(loader)
        roc_auc /= len(loader)

        print("Accuracy:", acc)
        print("ROCAUC:", roc_auc)
        return {"acc": acc, "roc_auc": roc_auc}

    def fit(self, train_data, best_parameters):
        """
        Trains a BertClassifier on train_data using a set of parameters.

        :param pd.DataFrame train_data: data on which the model has to be trained
        :param dict best_parameters: a dictionary containing the accuracy, prediction,
        and recall score of the models predictions on the data
        :return: The trained model
        """
        n_epochs = best_parameters["n_epochs"]
        lr = best_parameters["lr"]
        device = best_parameters["device"]

        train_data, val_data = train_test_split(train_data, test_size=0.2)
        train_data.index = range(len(train_data))
        val_data.index = range(len(val_data))

        train_loader = self.preprocess(data=train_data, parameters=best_parameters)["loader"]
        model = BertClassifier().to(device)
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        loss_func = nn.BCELoss()

        # train loop
        for epoch in range(1, n_epochs + 1):
            print("=== Epoch", epoch, "/", n_epochs, "===")
            model.train()
            for batch in train_loader:
                x_batch, y_batch, attention_mask = batch
                # model(x) = model.__call__(x) performs forward (+ more)
                probas = torch.flatten(model(x=x_batch, attention_mask=attention_mask))
                model.zero_grad()  # reset gradients from last step
                batch_loss = loss_func(probas, y_batch)  # calculate loss
                batch_loss.backward()  # calculate gradients
                optimizer.step()  # update parameters

            print("Metrics on training data after epoch", epoch, ":")
            self.predict(model=model, data=train_data, parameters=best_parameters)
            print("Metrics on validation data after epoch", epoch, ":")
            self.predict(model=model, data=val_data, parameters=best_parameters)
            print("\n")
        return {"model": model}


# read the datasets
folds = tools.read_folds(prefix="undersampled_stopped_text", read_path="../../data/folds_nlp", test_fold_id=0)
train_folds = folds["train"]
test_fold = folds["test"]
train_data = train_folds[0]
for i in range(1, len(train_folds) - 1):
    pd.concat([train_data, train_folds[i]], axis=0)

# define the parameters
device = tools.select_device()
print("device:", device)
parameters1 = tools.parameters_bert_based(n_epochs=4,
                                          lr=2e-5,
                                          max_seq_len=16,
                                          batch_size=32,
                                          x_name="text",
                                          y_name="label",
                                          device=device)

parameter_combinations = [parameters1]

# use the model
bert_wrapper = BertWrapper()
tools.performance_comparison(parameter_combinations=parameter_combinations,
                             wrapper=bert_wrapper,
                             folds=train_folds,
                             model_name="BERT")
best_bert_clf = bert_wrapper.fit(train_data=train_data, best_parameters=parameters1)["model"]
print("\nPERFORMANCE ON TEST:")
bert_wrapper.predict(model=best_bert_clf, data=test_fold, parameters=parameters1)
