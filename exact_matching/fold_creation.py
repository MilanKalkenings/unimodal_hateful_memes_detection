import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random_state = 0


class ExactCreator:
    """
    A class to create datasets from the given meme data.
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

    def create_regular(self, detected_share):
        """
        Creates the desired datasets. Creates train, val and test, as well as detected, non_detected
        balanced and imbalanced.
        The sets train, val and test can be utilized by classifier-based methods.
        The balanced dataset is a balanced concatenation of detected and unknown memes,
        whereas the imbalanced dataset is a concatenation of all unknown and known memes.

        :param float detected_share: defines the share of hateful memes that are assumed to be detected
        (and thus stored in our database) already
        """
        data = self.data.copy()
        hateful = data.loc[data["label"] == 1, :]
        detected = pd.DataFrame(hateful.sample(frac=1).head(int(len(hateful) * detected_share))["img"])
        detected["detected"] = [1] * len(detected)
        non_detected_index = [idx for idx in data.index if idx not in detected.index]
        non_detected = data.loc[non_detected_index, ["img"]]
        non_detected["detected"] = [0] * len(non_detected)

        non_detected_train, non_detected_test = train_test_split(non_detected, test_size=0.3)  # extract test
        non_detected_train, non_detected_val = train_test_split(non_detected_train, test_size=0.3)  # extract val

        train = pd.concat([non_detected_train, detected], axis=0).sample(frac=1)
        val = pd.concat([non_detected_val, detected], axis=0).sample(frac=1)
        test = pd.concat([non_detected_test, detected], axis=0).sample(frac=1)
        balanced = pd.concat([detected, non_detected.head(len(detected))], axis=0)  # 50 / 50
        imbalanced = pd.concat([detected, non_detected], axis=0)

        # write the datasets
        train.to_csv(self.destination_path + "exact_train_" + str(detected_share) + ".csv", index=False)
        val.to_csv(self.destination_path + "exact_val_" + str(detected_share) + ".csv", index=False)
        test.to_csv(self.destination_path + "exact_test_" + str(detected_share) + ".csv", index=False)
        balanced.to_csv(self.destination_path + "exact_balanced_" + str(detected_share) + ".csv", index=False)
        imbalanced.to_csv(self.destination_path + "exact_imbalanced_" + str(detected_share) + ".csv", index=False)

        detected.to_csv(self.destination_path + "exact_detected_" + str(detected_share) + ".csv", index=False)
        non_detected.to_csv(self.destination_path + "exact_non_detected_" + str(detected_share) + ".csv", index=False)


exact_creator = ExactCreator(train_path="../../data/hateful_memes_data/train.jsonl",
                             val_path="../../data/hateful_memes_data/dev.jsonl",
                             destination_path="../../data/exact_matching/")

exact_creator.create_regular(detected_share=0.05)
exact_creator.create_regular(detected_share=0.2)
