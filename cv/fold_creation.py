import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

np.random_state = 0


class ImageFoldCreator:
    """
    A class to create folds from the given cv data.
    """

    def __init__(self, train_path, val_path, destination_path):
        """
        Constructor.
        Mixes both, the original training- and validation data to obtain one large dataset to create folds from.
        :param str train_path: path of the original train data
        :param str val_path: path of the original validation data
        :param str destination_path: path of the destination folder
        """
        train = pd.read_json(train_path, lines=True)
        val = pd.read_json(val_path, lines=True)
        data = pd.concat([train, val], axis=0, ignore_index=True).loc[:, ["img", "label"]]
        data = data.sample(frac=1)  # shuffle
        data.index = range(len(data))
        self.data = data
        self.destination_path = destination_path

    @staticmethod
    def get_fold_ids(num_indices, num_folds, shuffle=True):
        """
        Creates <num_folds> many equal sized (as good as possible) sets of drawn indices.
        :param int num_indices: total number of indices to distribute.
        :param int num_folds: each set of indices is later on used to create one fold. The number of folds
        determines the number of sets on which the indices have to be distributed
        :param bool shuffle: determines if the indices have to be drawn randomly or not. Shuffle=True leads to
        randomly drawn sets of indices of (as good as possible) equal size.
        :return: a list containing num_folds many lists of indices.
        """
        indices = np.arange(num_indices)
        if shuffle:
            np.random.shuffle(indices)

        fold_indices = []
        last_border = 0
        for fold in range(1, num_folds + 1):
            border = int(1 / num_folds * fold * len(indices))
            fold_indices.append(indices[last_border:border])
            last_border = border
        return fold_indices

    def create_regular(self, prefix="img", num_folds=6):
        """
        Creates folds without undersampling.
        :param str prefix: a concatenation of the prefix and the fold number is used as the name of the .csv file
        in which the respective folds will be stored
        :param int num_folds: number of created folds.
        """
        fold_ids = self.get_fold_ids(num_indices=len(self.data), num_folds=num_folds)
        for i, fold in enumerate(fold_ids):
            fold = pd.DataFrame(self.data.iloc[fold, :])
            fold.to_csv(self.destination_path + prefix + str(i) + ".csv", index=False)

    def create_undersampled(self, prefix="undersampled_img", num_folds=6):
        """
        Creates an undersampled dataset having the same amount of members per category in the target variable.
        :param str prefix: a concatenation of the prefix and the fold number is used as the name of the .csv file
        in which the respective folds will be stored
        :param int num_folds: number of created folds.
        """
        goal = self.data["label"].value_counts().min()  # undersampling goal: equal amount of observations per class
        data_pos = self.data.loc[self.data["label"] == 1, :]
        data_neg = self.data.loc[self.data["label"] == 0, :]
        data = pd.concat([data_pos.head(goal), data_neg.head(goal)], axis=0, ignore_index=True)
        fold_ids = self.get_fold_ids(num_indices=len(data), num_folds=num_folds)
        for i, fold in enumerate(fold_ids):
            fold = pd.DataFrame(data.iloc[fold, :])
            fold.to_csv(self.destination_path + prefix + str(i) + ".csv", index=False)


image_fold_creator = ImageFoldCreator(train_path="../../data/hateful_memes_data/train.jsonl",
                                      val_path="../../data/hateful_memes_data/dev.jsonl",
                                      destination_path="../../data/folds_cv/")

image_fold_creator.create_regular()
image_fold_creator.create_undersampled()
