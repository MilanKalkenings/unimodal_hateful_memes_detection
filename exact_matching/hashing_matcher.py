import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torchvision import transforms

import tools


def dhash(img_path, transform, resize_goal=8, bidim=True):
    """
    Performs a variant of DHash to represent images by the difference between the pixel intensity along the rows.
    My implementation is vastly based on the ideas delineated in
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
    last access: 21.06.2021

    :param str img_path: path of the image that has to be hashed
    :param int resize_goal: width and height to which the image has to be reshaped
    :param bool test: a boolean determining whether transformations should be applied on the image or not
    :return:
    """
    image = Image.open("../../data/hateful_memes_data/" + img_path, formats=["PNG"]).convert("RGB")
    resize = transforms.Resize([resize_goal, resize_goal])
    image = resize(image)

    # transformations, in case the image is cropped / colored slightly different to the original
    if transform:
        color_jitter = transforms.ColorJitter(brightness=[0, 2],
                                              contrast=[0, 2],
                                              saturation=[0, 2],
                                              hue=[-0.1, 0.1])
        image = color_jitter(image)
        random_crop = transforms.RandomCrop(size=[resize_goal, resize_goal], pad_if_needed=True)
        image = random_crop(image)

    grayscale = transforms.Grayscale()
    image = grayscale(image)
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image).squeeze().numpy()

    if not bidim:  # check rows only
        diff = tensor[:, :-1] < tensor[:, 1:]  # is value i smaller than value i + 1?
        diff_flat = diff.astype(int).flatten()
    else:
        diff_row = tensor[:, :-1] < tensor[:, 1:]
        diff_row_flat = diff_row.astype(int).flatten()
        diff_col = tensor[:-1, :] < tensor[1:, :]
        diff_col_flat = diff_col.astype(int).flatten()
        diff_flat = np.concatenate([diff_row_flat, diff_col_flat])
    return diff_flat


def hash_difference(hashed_img1, hashed_img2, thresh):
    """
    Computes the difference between two hashed images.

    :param hashed_img1: the first hashed image
    :param hashed_img2: the second hashed image
    :return: the overall difference of the two hashed images
    """
    sum = 0
    for i in range(len(hashed_img1)):
        if hashed_img1[i] != hashed_img2[i]:
            sum += 1
        if sum > thresh:  # speed up
            break
    return sum


def detect(hash_diff, thresh):
    """
    Determines whether a hashed image should be considered as already detected given the difference between the 
    image and an already detected image in the database. The threshold can be interpreted as a slack variable allowing 
    uncertainties.
    
    :param int hash_diff: the difference between the image and an already detected image in the database
    :param int thresh: the amount of uncertainty. The higher this value, the more deviation from the already detected
    images still leads to a classification as 'already detected'
    :return: either 1 (already detected) or 0 (not detected)
    """
    if hash_diff < thresh:
        return 1
    return 0


def predict(data, detected, thresh):
    """
    Predicts whether memes in a dataset are already known to be hateful, and evaluates the results using
    accuracy, precision, and recall score.

    :param pd.DataFrame data: a DataFrame containing the memes that have to be classified
    :param pd.DataFrame detected: a DataFrame containing the memes that are known to be hateful
    :param int thresh: a threshold that determines how similar a hashed meme has to be to hashed detected
    hateful memes to be classified as 'known to be hateful'
    """
    detected_hash = detected["img"].apply(func=dhash, transform=False)
    for i in range(20):
        temp = detected["img"].apply(func=dhash, transform=True)
        detected_hash = pd.concat([detected_hash, temp])
        detected_hash.index = np.arange(start=0, stop=len(detected_hash))

    data_hash = data["img"].apply(func=dhash, transform=True)
    data_hash.name = "hash"
    data = pd.concat([data, data_hash], axis=1)

    predictions = []
    for i in range(len(data)):
        prediction = 0
        for j in range(len(detected_hash)):
            diff = hash_difference(data["hash"][i], detected_hash[j], thresh=thresh)
            prediction = detect(hash_diff=diff, thresh=thresh)
            if prediction == 1:
                break
        predictions.append(prediction)

    predictions = np.array(predictions)
    print("Accuracy:", accuracy_score(y_true=data["detected"], y_pred=predictions))
    print("Recall:", recall_score(y_true=data["detected"], y_pred=predictions, zero_division=0))
    print("Precision:", precision_score(y_true=data["detected"], y_pred=predictions, zero_division=0))


# read the data
data = tools.read_data(detected_share=0.05)
detected = data["detected"]
balanced = data["balanced"]
imbalanced = data["imbalanced"]

print(len(detected))
print(len(imbalanced))
print(len(balanced))

# perform the match-check:
for thresh in np.arange(start=3, stop=16, step=1):
    print("Balanced Dataset (50% detected, 50% unknown):", thresh)
    predict(data=balanced, detected=detected, thresh=thresh)
    print("Highly Imbalanced Dataset (~ 1.8% detected, ~98.2% unknown):")
    predict(data=imbalanced, detected=detected, thresh=thresh)
    print("\n")
