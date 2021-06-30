import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk import TweetTokenizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW

import tools


def read_glove_embedding(glove_path):
    """
    Reads the glove embedding from a .txt file, and creates a glove-vocabulary from it

    :param str glove_path: the relative path leading to the .txt file in which the embedding is stored
    :return: a pd.Series object having the words as indices and teh respective embedding-vectors as values
    """
    glove_map = {}
    with open(os.path.join(glove_path), encoding="utf-8") as file:
        for line in file:
            split = line.split()
            token = split[0]
            embedding = np.array(split[1:], dtype="float32")
            glove_map[token] = embedding
    glove_map = pd.Series(glove_map)
    return glove_map


class LSTMGloveClassifier(nn.Module):
    """
    An LSTM using pretrained word embeddings (glove)
    """

    def __init__(self, feats_per_time_step, hidden_size, n_layers, n_classes):
        """
        Constructor.

        :param int feats_per_time_step: each time step, i.e. each word is represented by a number of features.
        In case of word embeddings, the number of features per word is the embedding size of the word-vector.
        :param int hidden_size: size of the hidden state
        :param int n_layers: number of lstm layers
        :param int n_classes: determines how many classes have to be handled. 2 in binary case.
        """
        super(LSTMGloveClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(feats_per_time_step, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)  # initial hidden state
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)  # initial cell state
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.linear(out)
        return out


class BiLSTMGloveClassifier(nn.Module):
    """
    A bidirectional LSTM using pretrained word embeddings (glove)
    """

    def __init__(self, feats_per_time_step, hidden_size, n_layers, n_classes):
        """
        Constructor.

        :param int feats_per_time_step: each time step, i.e. each word is represented by a number of features.
        In case of word embeddings, the number of features per word is the embedding size of the word-vector.
        :param int hidden_size: size of the hidden state
        :param int n_layers: number of lstm layers
        :param int n_classes: determines how many classes have to be handled. 2 in binary case.
        """
        super(BiLSTMGloveClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(feats_per_time_step, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, n_classes)

    def forward(self, x):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size).to(device)  # initial hidden state
        c0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size).to(device)  # initial cell state
        out, _ = self.lstm(x, (h0, c0))
        out_direction1 = out[:, -1, :self.hidden_size]  # regular
        out_direction2 = out[:, 0, self.hidden_size:]  # reverse
        out = torch.cat((out_direction1, out_direction2), 1)
        out = self.linear(out)
        return out


class GloveWrapper:

    def __init__(self, glove_map, glove_size):
        """
        Constructor.

        :param pd.Series glove_map: a pd.Series having words as indices and word-vectors as values.
        Maps words to their (glove) embeddings.
        :param int glove_size: glove embeddings come in different sizes (50, 100, 200, 300), issue the used size
        in this parameter.
        """
        self.glove_map = glove_map
        self.glove_size = glove_size

    def preprocess(self, data, parameters):
        """
        Preprocesses the data of a fold and returns the DataLoaders for the wrapped neural network.

        :param pd.DataFrame data: one fold of data to be processed. Contains a column <x_name> containing text
        sequences and another column <y_name> containing the class labels of the sequence
        :param dict parameters: a dictionary containing the parameters defined in tools.parameters_rnn_based
        :return: a dictionary having the key "loader" and the constructed DataLoader as value.
        """
        max_seq_len = parameters["max_seq_len"]
        batch_size = parameters["batch_size"]
        x_name = parameters["x_name"]
        y_name = parameters["y_name"]
        device = parameters["device"]
        text_col = data[x_name]
        target_col = data[y_name]
        tweet_tokenizer = TweetTokenizer()

        def tokenize_sequence(sequence):
            """
            Tokenizes a sequence using the nltk.TweetTokenizer

            :param str sequence: a sequence of text
            :return: the tokenized sequence
            """
            return tweet_tokenizer.tokenize(sequence)

        def tokens_to_embedding(token_list):
            """
            Embeds a list of tokens.

            :param list token_list: a list of strings. Each string is one token.
            :return: a numpy array of embedded tokens using the glove word embedding
            """
            embedding = []
            for i, token in enumerate(token_list):
                if i == max_seq_len:
                    break
                if token in glove_map.index:
                    embedding.append(glove_map[token])
                else:
                    embedding.append([0] * self.glove_size)
            padding_len = max_seq_len - len(embedding)
            for i in range(padding_len):
                embedding.append(np.zeros(self.glove_size))
            return np.stack(embedding)

        tokenized_texts = text_col.apply(func=tokenize_sequence)
        embedded_texts = tokenized_texts.apply(func=tokens_to_embedding)
        x = torch.tensor(embedded_texts, dtype=torch.float32).to(device)
        y = torch.tensor(target_col, dtype=torch.long).to(device)  # long for CrossEntropyLoss
        dataset = TensorDataset(x, y)
        sampler = RandomSampler(dataset)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
        return {"loader": loader}

    def fit(self, train_data, best_parameters):
        """
        Trains a (bi)LSTMGloveClassifier on train_data using a set of parameters.

        :param pd.DataFrame train_data: data on which the model has to be trained
        :param dict best_parameters: a dictionary containing the parameters defined in tools.parameters_rnn_based
        :return: The trained model
        """
        n_epochs = best_parameters["n_epochs"]
        lr = best_parameters["lr"]
        n_layers = best_parameters["n_layers"]
        feats_per_time_step = best_parameters["feats_per_time_step"]
        hidden_size = best_parameters["hidden_size"]
        n_classes = best_parameters["n_classes"]
        device = best_parameters["device"]

        preprocessed = self.preprocess(data=train_data, parameters=best_parameters)

        train_loader = preprocessed["loader"]

        model = LSTMGloveClassifier(feats_per_time_step=feats_per_time_step,
                                    hidden_size=hidden_size,
                                    n_layers=n_layers,
                                    n_classes=n_classes).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        loss_func = nn.CrossEntropyLoss()

        # train loop
        for epoch in range(1, n_epochs + 1):
            print("=== Epoch", epoch, "/", n_epochs, "===")
            model.train()
            for i, batch in enumerate(train_loader):
                x_batch, y_batch = batch
                probas = model(x=x_batch)  # model(x) = model.__call__(x) performs forward (+ more)
                model.zero_grad()  # reset gradients from last step
                batch_loss = loss_func(probas, y_batch)  # calculate loss
                batch_loss.backward()  # calculate gradients
                optimizer.step()  # update parameters

            print("Metrics on training data after epoch", epoch, ":")
            self.predict(model=model, data=train_data, parameters=best_parameters)
        return {"model": model}

    def predict(self, model, data, parameters):
        """
        Predicts the labels of a dataset and evaluates the results against the ground truth.

        :param CNNClassifier model: a trained CNNClassifier
        :param pd.DataFrame data: a dataset on which the prediction has to be performed
        :param dict parameters: a dictionary containing the parameters defined in tools.parameters_cnn
        :return: a dictionary containing the accuracy and roc-auc score of the models predictions on the data
        """
        model.eval()
        acc = 0
        roc_auc = 0
        loader = self.preprocess(data=data, parameters=parameters)["loader"]
        for batch in loader:
            x_batch, y_batch = batch
            with torch.no_grad():
                probas = model(x=x_batch)
            _, preds = torch.max(probas.data, 1)
            metrics = tools.evaluate(y_true=y_batch, y_probas=preds)
            acc += metrics["acc"]
            roc_auc += metrics["roc_auc"]

        acc /= len(loader)
        roc_auc /= len(loader)

        print("Accuracy:", acc)
        print("ROCAUC:", roc_auc)
        return {"acc": acc, "roc_auc": roc_auc}

    def evaluate_hyperparameters(self, folds, parameters):
        """
        Evaluates the given parameters on multiple folds using k-fold cross validation.

        :param list folds: a list of pd.DataFrames. Each of the DataFrames contains one fold of the data available
        during the training time.
        :param dict parameters: a dictionary containing the parameters defined in tools.parameters_rnn_based
        :return: a dictionary containing the accuracy and roc-auc scores on both training and validation data
        """
        n_epochs = parameters["n_epochs"]
        lr = parameters["lr"]
        n_layers = parameters["n_layers"]
        feats_per_time_step = parameters["feats_per_time_step"]
        hidden_size = parameters["hidden_size"]
        n_classes = parameters["n_classes"]
        device = parameters["device"]

        acc_scores_train = np.zeros(n_epochs)
        roc_auc_scores_train = np.zeros(n_epochs)

        acc_scores = np.zeros(n_epochs)
        roc_auc_scores = np.zeros(n_epochs)

        loss_func = nn.CrossEntropyLoss()
        for fold_id in range(len(folds)):
            print("=== Fold", fold_id + 1, "/", len(folds), "===")
            sets = tools.train_val_split(data_folds=folds, val_fold_id=fold_id)
            train = sets["train"]
            val = sets["val"]
            train_loader = self.preprocess(data=train, parameters=parameters)["loader"]
            model = LSTMGloveClassifier(feats_per_time_step=feats_per_time_step,
                                        hidden_size=hidden_size,
                                        n_layers=n_layers,
                                        n_classes=n_classes).to(device)  # isolated model per fold
            optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

            for epoch in range(1, n_epochs + 1):
                print("=== Epoch", epoch, "/", n_epochs, "===")
                model.train()
                for i, batch in enumerate(train_loader):
                    x_batch, y_batch = batch
                    probas = model(x=x_batch)  # forward
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


# read the data
glove_map = read_glove_embedding(glove_path="../../data/pretrained_embeddings/glove.6B.50d.txt")
folds = tools.read_folds(prefix="undersampled_stopped_text", read_path="../../data/folds_nlp", test_fold_id=0)
train_folds = folds["train"]
test_fold = folds["test"]
train_data = train_folds[0]
for i in range(1, len(train_folds) - 1):
    pd.concat([train_data, train_folds[i]], axis=0)

# define the parameters
glove_size = 50
device = tools.select_device()
print("device:", device)
parameters1 = tools.parameters_rnn_based(n_epochs=10,
                                         lr=0.001,
                                         max_seq_len=16,
                                         n_layers=3,
                                         feats_per_time_step=glove_size,
                                         hidden_size=16,
                                         n_classes=2,
                                         batch_size=64,
                                         x_name="text",
                                         y_name="label",
                                         device=device)

parameters2 = tools.parameters_rnn_based(n_epochs=10,
                                         lr=0.0001,
                                         max_seq_len=16,
                                         n_layers=3,
                                         feats_per_time_step=glove_size,
                                         hidden_size=16,
                                         n_classes=2,
                                         batch_size=64,
                                         x_name="text",
                                         y_name="label",
                                         device=device)

parameters3 = tools.parameters_rnn_based(n_epochs=10,
                                         lr=0.001,
                                         max_seq_len=14,
                                         n_layers=3,
                                         feats_per_time_step=glove_size,
                                         hidden_size=16,
                                         n_classes=2,
                                         batch_size=64,
                                         x_name="text",
                                         y_name="label",
                                         device=device)

parameters4 = tools.parameters_rnn_based(n_epochs=10,
                                         lr=0.001,
                                         max_seq_len=16,
                                         n_layers=3,
                                         feats_per_time_step=glove_size,
                                         hidden_size=128,
                                         n_classes=2,
                                         batch_size=64,
                                         x_name="text",
                                         y_name="label",
                                         device=device)

parameters5 = tools.parameters_rnn_based(n_epochs=10,
                                         lr=0.001,
                                         max_seq_len=16,
                                         n_layers=3,
                                         feats_per_time_step=glove_size,
                                         hidden_size=16,
                                         n_classes=2,
                                         batch_size=32,
                                         x_name="text",
                                         y_name="label",
                                         device=device)

parameter_combinations = [parameters1, parameters2, parameters3, parameters4, parameters5]

# use the model
lstmg_wrapper = GloveWrapper(glove_map=glove_map, glove_size=glove_size)
tools.performance_comparison(parameter_combinations=parameter_combinations,
                             wrapper=lstmg_wrapper,
                             folds=train_folds,
                             model_name="GloVe LSTM")
fitted = lstmg_wrapper.fit(train_data=train_data, best_parameters=parameters1)
best_lstmg_clf = fitted["model"]
print("\nPERFORMANCE ON TEST:")
lstmg_wrapper.predict(model=best_lstmg_clf, data=test_fold, parameters=parameters1)
