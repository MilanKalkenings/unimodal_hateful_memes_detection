import pandas as pd
import numpy as np
import os
import shutil

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

    def create_regular(self, img_folder="../../data/hateful_memes_data/img", prefix="img", x_name="img", y_name="label"):
        """
        Creates folds without any further transformations of undersampling.

        :param str prefix: a concatenation of the prefix and the fold number is used as the name of the .csv file
        in which the respective folds are stored
        """
        def distribute_images(row, pos_path, neg_path):
            if row[y_name] == 0:  # non-hateful memes are of the negative class
                source_path = img_folder + "/" + row[x_name]
                shutil.copy(src=source_path, dst=neg_path + row[x_name])
            else:
                source_path = img_folder + "/" + row[x_name]
                shutil.copy(src=source_path, dst=pos_path + row[x_name])

        def clean_path(path):
            return path[3:]

        fold_ids = self.get_fold_ids(num_indices=len(self.data), num_folds=6)
        for i, fold in enumerate(fold_ids):
            print("handling fold", i+1)
            fold_path = self.destination_path + "/fold" + str(i)
            pos_path = fold_path + "/" + prefix + "_pos" + str(i)
            neg_path = fold_path + "/" + prefix + "_neg" + str(i)
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
            if not os.path.exists(pos_path):
                os.makedirs(pos_path)
            if not os.path.exists(neg_path):
                os.makedirs(neg_path)

            fold = pd.DataFrame(self.data.iloc[fold, :])
            fold["img"] = fold["img"].apply(func=clean_path)
            fold.apply(func=distribute_images, pos_path=pos_path, neg_path=neg_path, axis=1)

    def create_custom(self, prefix="img"):
        fold_ids = self.get_fold_ids(num_indices=len(self.data), num_folds=6)
        for i, fold in enumerate(fold_ids):
            fold = pd.DataFrame(self.data.iloc[fold, :])
            fold.to_csv(self.destination_path + prefix + str(i) + ".csv", index=False)

image_fold_creator = ImageFoldCreator(train_path="../../data/hateful_memes_data/train.jsonl",
                                      val_path="../../data/hateful_memes_data/dev.jsonl",
                                      destination_path="../../data/folds_cv/")
image_fold_creator.create_custom()
