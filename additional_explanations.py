from sklearn.metrics import f1_score
import numpy as np


def f1_demo(y_true, y_pred):
    """
    Demonstrates the f1 score.

    :param np.array y_true: some labels
    :param np.array y_pred: some predicted labels
    """
    print("given the true labels\n", y_true, "\nand the predicted labels\n", y_pred, "\nthe f1-score is")
    print(f1_score(y_true=y_true, y_pred=y_pred), "\n")


f1_demo(y_true=np.array([0, 0, 0, 0, 1, 1, 0]), y_pred=np.array([0, 0, 0, 0, 0, 0, 0]))
f1_demo(y_true=np.array([0, 0, 0, 0, 1, 1, 0]), y_pred=np.array([1, 1, 0, 0, 0, 0, 1]))
f1_demo(y_true=np.array([0, 0, 0, 0, 1, 1, 0]), y_pred=np.array([0, 0, 0, 0, 0, 1, 0]))
f1_demo(y_true=np.array([0, 0, 0, 0, 1, 1, 0]), y_pred=np.array([0, 0, 0, 0, 0, 1, 1]))
f1_demo(y_true=np.array([0, 0, 0, 0, 1, 1, 0]), y_pred=np.array([0, 1, 0, 0, 0, 1, 1]))
f1_demo(y_true=np.array([0, 0, 0, 0, 1, 1, 0]), y_pred=np.array([0, 0, 0, 0, 1, 1, 1]))
f1_demo(y_true=np.array([0, 0, 0, 0, 1, 1, 0]), y_pred=np.array([0, 0, 0, 0, 1, 1, 0]))
f1_demo(y_true=np.array([0, 0, 0, 0, 1, 1, 0]), y_pred=np.array([1, 1, 1, 1, 1, 1, 1]))
