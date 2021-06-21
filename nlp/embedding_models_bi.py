from nltk import TweetTokenizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import pandas as pd
import numpy as np
from transformers import AdamW
import torch
import tools
import torch.nn as nn


class RNNEClassifier(nn.Module):
    """
    An RNN-based classifier that trains word embeddings itself.
    """

    def __init__(self, feats_per_time_step, hidden_size, n_layers, n_classes, vocab_size):
        """
        Constructor.

        :param int feats_per_time_step: each time step, i.e. each word is represented by a number of features.
        In case of word embeddings, the number of features per word is the embedding size of the word-vector.
        :param int hidden_size: size of the hidden state
        :param int n_layers: number of lstm layers
        :param int n_classes: determines how many classes have to be handled. 2 in binary case.
        :param int vocab_size: number of tokens identified during training.
        """
        super(RNNEClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # vocab size + padding_token + out_of_vocab_token, i.e. vocab_size+2
        self.emb = nn.Embedding(num_embeddings=vocab_size + 2, embedding_dim=feats_per_time_step, padding_idx=0)
        self.rnn = nn.RNN(feats_per_time_step, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        x = self.emb(x)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)  # size: batch_size x seq_len x hidden_size
        out = out[:, -1, :]
        out = self.linear(out)
        return out


class BiRNNEClassifier(nn.Module):
    """
    A bidirectional RNN-based classifier that trains word embeddings itself.
    """

    def __init__(self, feats_per_time_step, hidden_size, n_layers, n_classes, vocab_size):
        """
        Constructor.

        :param int feats_per_time_step: each time step, i.e. each word is represented by a number of features.
        In case of word embeddings, the number of features per word is the embedding size of the word-vector.
        :param int hidden_size: size of the hidden state
        :param int n_layers: number of lstm layers
        :param int n_classes: determines how many classes have to be handled. 2 in binary case.
        :param int vocab_size: number of tokens identified during training.
        """
        super(BiRNNEClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # vocab size + padding_token + out_of_vocab_token, i.e. vocab_size+2
        self.emb = nn.Embedding(num_embeddings=vocab_size + 2, embedding_dim=feats_per_time_step, padding_idx=0)
        self.rnn = nn.RNN(feats_per_time_step, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, n_classes)

    def forward(self, x):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        x = self.emb(x)
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)  # size: batch_size x seq_len x hidden_size
        out_direction1 = out[:, -1, :self.hidden_size]  # regular
        out_direction2 = out[:, 0, self.hidden_size:]  # reverse
        out = torch.cat((out_direction1, out_direction2), 1)
        out = self.linear(out)
        return out


class GRUEClassifier(nn.Module):
    """
    A GRU-based classifier that trains word embeddings itself.
    """

    def __init__(self, feats_per_time_step, hidden_size, n_layers, n_classes, vocab_size):
        """
        Constructor.

        :param int feats_per_time_step: each time step, i.e. each word is represented by a number of features.
        In case of word embeddings, the number of features per word is the embedding size of the word-vector.
        :param int hidden_size: size of the hidden state
        :param int n_layers: number of lstm layers
        :param int n_classes: determines how many classes have to be handled. 2 in binary case.
        :param int vocab_size: number of tokens identified during training.
        """
        super(GRUEClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # vocab size + padding_token + out_of_vocab_token, i.e. vocab_size+2
        self.emb = nn.Embedding(num_embeddings=vocab_size + 2, embedding_dim=feats_per_time_step, padding_idx=0)
        self.gru = nn.GRU(feats_per_time_step, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        x = self.emb(x)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.linear(out)
        return out


class BiGRUEClassifier(nn.Module):
    """
    A GRU-based classifier that trains word embeddings itself.
    """

    def __init__(self, feats_per_time_step, hidden_size, n_layers, n_classes, vocab_size):
        """
        Constructor.

        :param int feats_per_time_step: each time step, i.e. each word is represented by a number of features.
        In case of word embeddings, the number of features per word is the embedding size of the word-vector.
        :param int hidden_size: size of the hidden state
        :param int n_layers: number of lstm layers
        :param int n_classes: determines how many classes have to be handled. 2 in binary case.
        :param int vocab_size: number of tokens identified during training.
        """
        super(BiGRUEClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # vocab size + padding_token + out_of_vocab_token, i.e. vocab_size+2
        self.emb = nn.Embedding(num_embeddings=vocab_size + 2, embedding_dim=feats_per_time_step, padding_idx=0)
        self.gru = nn.GRU(feats_per_time_step, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, n_classes)

    def forward(self, x):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        x = self.emb(x)
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out_direction1 = out[:, -1, :self.hidden_size]  # regular
        out_direction2 = out[:, 0, self.hidden_size:]  # reverse
        out = torch.cat((out_direction1, out_direction2), 1)
        out = self.linear(out)
        return out


class LSTMEClassifier(nn.Module):
    """
    An LSTM-based classifier that trains word embeddings itself.
    """

    def __init__(self, feats_per_time_step, hidden_size, n_layers, n_classes, vocab_size):
        """
        Constructor.

        :param int feats_per_time_step: each time step, i.e. each word is represented by a number of features.
        In case of word embeddings, the number of features per word is the embedding size of the word-vector.
        :param int hidden_size: size of the hidden state
        :param int n_layers: number of lstm layers
        :param int n_classes: determines how many classes have to be handled. 2 in binary case.
        :param int vocab_size: number of tokens identified during training.
        """
        super(LSTMEClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # vocab size + padding_token + out_of_vocab_token, i.e. vocab_size+2
        self.emb = nn.Embedding(num_embeddings=vocab_size + 2, embedding_dim=feats_per_time_step, padding_idx=0)
        self.lstm = nn.LSTM(feats_per_time_step, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        x = self.emb(x)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.linear(out)
        return out


class BiLSTMEClassifier(nn.Module):
    """
    An LSTM-based classifier that trains word embeddings itself.
    """

    def __init__(self, feats_per_time_step, hidden_size, n_layers, n_classes, vocab_size):
        """
        Constructor.

        :param int feats_per_time_step: each time step, i.e. each word is represented by a number of features.
        In case of word embeddings, the number of features per word is the embedding size of the word-vector.
        :param int hidden_size: size of the hidden state
        :param int n_layers: number of lstm layers
        :param int n_classes: determines how many classes have to be handled. 2 in binary case.
        :param int vocab_size: number of tokens identified during training.
        """
        super(BiLSTMEClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # vocab size + padding_token + out_of_vocab_token, i.e. vocab_size+2
        self.emb = nn.Embedding(num_embeddings=vocab_size + 2, embedding_dim=feats_per_time_step, padding_idx=0)
        self.lstm = nn.LSTM(feats_per_time_step, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, n_classes)

    def forward(self, x):
        """
        performs the forward pass.

        :param torch.Tensor x: the input/observation per batch
        :return: the prediction of the whole batch
        """
        x = self.emb(x)
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out_direction1 = out[:, -1, :self.hidden_size]  # regular
        out_direction2 = out[:, 0, self.hidden_size:]  # reverse
        out = torch.cat((out_direction1, out_direction2), 1)
        out = self.linear(out)
        return out


class EmbeddingWrapper:
    """
    A wrapper for the textual classifiers, that enables the core functionalities of the network.
    """

    def __init__(self, model_class):
        """
        Constructor.

        :param nn.Module model_class: one of the text-classifiers in this python module.
        """
        self.model_class = model_class

    @staticmethod
    def preprocess(data, parameters, vocab=None):
        """
        Preprocesses the data of a fold and returns the DataLoaders for the wrapped neural network.

        :param pd.DataFrame data: one fold of data to be processed. Contains a column <x_name> containing text
        sequences and another column <y_name> containing the class labels of the sequence
        :param dict parameters: all parameters needed for the preprocessing.
        :param pd.Series vocab: the vocabulary handled for the preprocessing, if None, a new vocabulary for the whole
        textual information in data is created (create new one on train, use existing one for test/val)
        :return: a dictionary having the key "loader" and the constructed DataLoader as value. (dictionary to match the
        pattern of the project); If vocab=None, the key "vocab" is added to the dictionary. The value of that key
        is the extracted vocabulary of the whole data
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

        def generate_vocab(tokenized_train):
            """
            Generates the vocabulary on the whole tokenized textual data.

            :param pd.Series tokenized_train: a pd.Series having lists of separated tokens of the respective
            sequences/observations as values.
            :return: a pd.Series representing the vocabulary-mapping. The indices (1...N) are the token
            ids, and the values are the corresponding strings of the tokens.
            """
            all_tokens = []

            def collapse_sequences(row, all_tokens):
                """
                Combines lists of tokens to one large list of tokens.

                :param list row: a list of tokens stored in one row of the column having the textual data.
                :param list all_tokens: a list having all already concatenated tokens
                """
                all_tokens += row

            tokenized_train.apply(func=collapse_sequences, all_tokens=all_tokens)
            vocab = pd.Series(all_tokens, name="token").drop_duplicates()
            vocab.index = np.arange(start=1, stop=len(vocab) + 1)  # start with index 1, 0 will be used for padding
            return vocab

        def encode_tokens(tokenized_data, vocab):
            """
            Encodes token ids.

            :param pd.Series tokenized_data: a pd.Series containing the tokenized sequences. Each tokenized sequence is
            a list of strings
            :param pd.Series vocab: a pd.Series having token ids as indices and token strings as values.
            :return: a pd.Series of lists of encoded tokens. Each token is encoded by its own id
            """

            def encode_row(row):
                """
                Encodes one row/observation/sequence of tokenized textual data.
                Might truncate or pad the sequence.

                :param list row: a list of strings containing tokens from a sequence
                :return: the encoded row stored in a list
                """
                encoded_row = []
                for i, token in enumerate(row):
                    if token in list(vocab.values):
                        encoded_row.append(vocab.loc[vocab == token].index[0])
                    else:
                        encoded_row.append(len(vocab) + 1)  # out of vocabulary token
                    if i == max_seq_len - 1:
                        break  # truncate long sequences if necessary
                padding = [0] * (max_seq_len - len(encoded_row))
                return encoded_row + padding

            return tokenized_data.apply(func=encode_row)

        if vocab is None:
            x_tokens = text_col.apply(func=tokenize_sequence)
            vocab = generate_vocab(tokenized_train=x_tokens)
            x_encoded = encode_tokens(tokenized_data=x_tokens, vocab=vocab)
            x_encoded = torch.tensor(x_encoded, dtype=torch.long).to(device)
            y = torch.tensor(target_col, dtype=torch.long).to(device)  # long for CrossEntropyLoss
            dataset = TensorDataset(x_encoded, y)
            sampler = RandomSampler(dataset)
            loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
            return {"loader": loader, "vocab": vocab}
        else:
            x_tokens = text_col.apply(func=tokenize_sequence)
            x_encoded = encode_tokens(tokenized_data=x_tokens, vocab=vocab)
            x_encoded = torch.tensor(x_encoded, dtype=torch.long).to(device)
            y = torch.tensor(target_col, dtype=torch.long).to(device)
            dataset = TensorDataset(x_encoded, y)
            sampler = RandomSampler(dataset)
            loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
            return {"loader": loader}  # to preserve the pattern

    def fit(self, train_data, best_parameters, verbose=2):
        """
        Trains a textual classifier on train_data using a set of parameters.

        :param pd.DataFrame train_data: data on which the model has to be trained
        :param dict best_parameters: #TODO
        :param int verbose: defines the amount of prints made during the call. The higher, the more prints
        :return: The trained model
        """
        n_epochs = best_parameters["n_epochs"]
        lr = best_parameters["lr"]
        n_layers = best_parameters["n_layers"]
        feats_per_time_step = best_parameters["feats_per_time_step"]
        hidden_size = best_parameters["hidden_size"]
        n_classes = best_parameters["n_classes"]
        device = best_parameters["device"]

        preprocessed = self.preprocess(data=train_data, parameters=parameters)
        train_loader = preprocessed["loader"]
        vocab = preprocessed["vocab"]
        model = self.model_class(feats_per_time_step=feats_per_time_step,
                                 hidden_size=hidden_size,
                                 n_layers=n_layers,
                                 n_classes=n_classes,
                                 vocab_size=len(vocab)).to(device)
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

            if verbose > 0:
                print("Metrics on training data after epoch", epoch, ":")
                self.predict(model=model, data=train_data, parameters=best_parameters, vocab=vocab)
        return {"model": model, "vocab": vocab}

    def evaluate_hyperparameters(self, folds, parameters):
        """
        Evaluates the given parameters on multiple folds using k-fold cross validation.

        :param list folds: a list of pd.DataFrames. Each of the DataFrames contains one fold of the data available
        during the training time.
        :param dict parameters: #TODO
        :return: a dictionary having the keys "acc_scores", "f1_scores" and "parameters", having the accuracy score
        for each fold, the f1 score of each fold and the used parameters as values
        """
        device = parameters["device"]
        n_epochs = parameters["n_epochs"]
        lr = parameters["lr"]
        n_layers = parameters["n_layers"]
        feats_per_time_step = parameters["feats_per_time_step"]
        hidden_size = parameters["hidden_size"]
        n_classes = parameters["n_classes"]

        acc_scores = np.zeros(n_epochs)
        f1_scores = np.zeros(n_epochs)
        loss_func = nn.CrossEntropyLoss()
        for fold_id in range(len(folds)):
            print("=== Fold", fold_id + 1, "/", len(folds), "===")
            sets = tools.train_val_split(data_folds=folds, val_fold_id=fold_id)
            train = sets["train"]
            val = sets["val"]
            preprocessed = self.preprocess(data=train, parameters=parameters)
            train_loader = preprocessed["loader"]
            vocab = preprocessed["vocab"]
            model = self.model_class(feats_per_time_step=feats_per_time_step,
                                     hidden_size=hidden_size,
                                     n_layers=n_layers,
                                     n_classes=n_classes,
                                     vocab_size=len(vocab)).to(device)  # isolated model per fold
            optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)  # depends on model

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
                self.predict(model=model, data=train, parameters=parameters, vocab=vocab)
                print("Metrics on validation data after epoch", epoch, ":")
                metrics = self.predict(model=model, data=val, parameters=parameters, vocab=vocab)
                acc_scores[epoch - 1] += metrics["acc"]
                f1_scores[epoch - 1] += metrics["f1"]
                print("\n")

        for i in range(n_epochs):
            acc_scores[i] /= len(folds)
            f1_scores[i] /= len(folds)
        return {"acc_scores": acc_scores, "f1_scores": f1_scores, "parameters": parameters}

    def predict(self, model, data, parameters, vocab):
        """
        Predicts the labels of a dataset and evaluates the results against the ground truth.

        :param nn.Module model: a trained text classifier
        :param pd.DataFrame data: a dataset on which the prediction has to be performed
        :param dict parameters: a dictionary having at least the keys "max_seq_len", "batch_size", "x_name", "y_name",
        "device", and the respective values.
        :param pd.Series vocab: a trained vocab mapping that maps tokens to integers
        :return: a dictionary containing the f1_score and the accuracy_score of the models predictions on the data
        """
        model.eval()
        acc = 0
        f1 = 0
        precision = 0
        recall = 0
        loader = self.preprocess(data=data, parameters=parameters, vocab=vocab)["loader"]
        for batch in loader:
            x_batch, y_batch = batch
            with torch.no_grad():
                probas = model(x=x_batch)
            _, preds = torch.max(probas.data, 1)
            metrics = tools.evaluate(y_true=y_batch, y_probas=preds)
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
folds = tools.read_folds(prefix="undersampled_stopped_text",
                         read_path="../../data/folds_nlp",
                         test_fold_id=0)
train_folds = folds["train"]
test_fold = folds["test"]
train_data = train_folds[0]
for i in range(1, len(train_folds) - 1):
    pd.concat([train_data, train_folds[i]], axis=0)

# define the parameters
device = tools.select_device()
print("device:", device)
parameters = {"n_epochs": 5,
              "lr": 0.001,
              "max_seq_len": 16,
              "n_layers": 3,
              "feats_per_time_step": 128,
              "hidden_size": 16,
              "n_classes": 2,
              "batch_size": 32,
              "x_name": "text",
              "y_name": "label",
              "device": device}

# use the model
e_wrapper = EmbeddingWrapper(model_class=BiLSTMEClassifier)
#print(e_wrapper.evaluate_hyperparameters(folds=train_folds, parameters=parameters))
fitted = e_wrapper.fit(train_data=train_data, best_parameters=parameters, verbose=1)
vocab = fitted["vocab"]
best_e_clf = fitted["model"]
print("\nPERFORMANCE ON TEST:")
e_wrapper.predict(model=best_e_clf, data=test_fold, parameters=parameters, vocab=vocab)
