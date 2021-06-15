import tools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW
from transformers import BertModel
from transformers import BertTokenizer


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
        self.linear = nn.Linear(768, 1)  # input of the first custom layer has to match dim of the BERT-output
        self.sigmoid = nn.Sigmoid()  # activation function applied to obtain probabilities

    def forward(self, x, attention_mask):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        bert_output = self.bert(input_ids=x, attention_mask=attention_mask)
        linear_output = self.linear(bert_output[1])
        probas = self.sigmoid(linear_output)
        return probas


class BertWrapper:

    @staticmethod
    def preprocess(data, parameters):
        """
        Preprocesses the data of a fold and returns the DataLoaders for the wrapped neural network.

        :param pd.DataFrame data: one fold of data to be processed. Contains a column <x_name> containing text
        sequences and another column <y_name> containing the class labels of the sequence
        :param int max_seq_len: maximum length of a sequence. Shorter sequences will be zero-padded to this size,
        longer sequences will be truncated to this size
        :param int batch_size: number of observations handled in each batch
        :param str x_name: name of the column containing the text-sequences
        :param str y_name: name of the column containing the class labels
        :param str device: name of the device (usually "cuda" or "cpu")
        :return: a dictionary having the key "loader" and the constructed DataLoader as value. (dictionary to match the
        pattern of the project)
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
                                            add_special_tokens=True,  # necessary for BERT
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

    def evaluate_hyperparameters(self, folds, parameters, verbose=0):
        """
        Evaluates the given parameters on multiple folds using k-fold cross validation.

        :param list folds: a list of pd.DataFrames. Each of the DataFrames contains one fold of the data available
        during the training time.
        :param dict parameters: a dictionary containing one combination of  parameters.
         The dictionary has at least the keys "n_epochs", "lr", "max_seq_len",
        "n_layers", "feats_per_time_step", "hidden_size", "n_classes", "batch_size", "x_name", "y_name", "device",
        and the respective values
        :param int verbose: defines the amount of prints made during the call. The higher, the more prints
        :return: a dictionary having the keys "acc_scores", "f1_scores" and "parameters", having the accuracy score
        for each fold, the f1 score of each fold and the used parameters as values
        """
        n_epochs = parameters["n_epochs"]
        lr = parameters["lr"]
        device = parameters["device"]

        acc_scores = np.zeros(n_epochs)
        f1_scores = np.zeros(n_epochs)
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

        :param BertClassifier model: a trained BERTClassifier
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
            x_batch, y_batch, attention_mask = batch
            with torch.no_grad():
                probas = torch.flatten(model(x=x_batch, attention_mask=attention_mask))
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

    def fit(self, train_data, best_parameters, verbose=0):
        """
        Trains a BertClassifier on train_data using a set of parameters.

        :param pd.DataFrame train_data: data on which the model has to be trained
        :param dict best_parameters: a dictionary having the keys "n_epochs", "lr", "max_seq_len", "batch_size",
        "x_name", "y_name", "device", and the respective values. The dictionary contains the best discovered
        hyperparameter combination found using evaluate_hyperparameters on another instance of this class.
        :param int verbose: defines how many prints are performed. The higher, the better for debugging.
        """
        n_epochs = best_parameters["n_epochs"]
        lr = best_parameters["lr"]
        device = best_parameters["device"]

        train_loader = self.preprocess(data=train_data, parameters=parameters)["loader"]
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

            if verbose > 0:
                print("Metrics on training data after epoch", epoch, ":")
                self.predict(model=model, data=train_data, parameters=best_parameters)
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
parameters = {"n_epochs": 1,
              "lr": 2e-5,
              "max_seq_len": 16,
              "batch_size": 32,
              "x_name": "text",
              "y_name": "label",
              "device": device}

# use the model
bert_wrapper = BertWrapper()
print(bert_wrapper.evaluate_hyperparameters(folds=train_folds, parameters=parameters))
best_bert_clf = bert_wrapper.fit(train_data=train_data, best_parameters=parameters, verbose=1)["model"]
print("\nPERFORMANCE ON TEST:")
bert_wrapper.predict(model=best_bert_clf, data=test_fold, parameters=parameters)
