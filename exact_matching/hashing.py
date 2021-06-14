import hashlib
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import io

class HashingMatcher:
    def __init__(self, data_path, detected_path, hash_func):
        self.data_path = data_path
        self.detected_path = detected_path
        self.hash_func = hash_func

    def img_path_to_hash(self, img_path):
        """

        :param str img_path:
        :return:
        """
        # path to tensor
        path = self.data_path + img_path
        transform_pipe = transforms.Compose([transforms.ToTensor()])

        image = Image.open(path, formats=["PNG"])
        tensor = transform_pipe(image)
        if tensor.size(0) == 4:  # very few images have one more channel, change to RGB format
            image = Image.open(path, formats=["PNG"]).convert("RGB")
            tensor = transform_pipe(image)

        # tensor to bytes
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        img_bytes = buffer.read()

        # bytes to hash
        return self.hash_func(img_bytes).hexdigest()

    def predict(self, hashed_image):
        detected = pd.read_csv(self.detected_path)["detected"]  # stored as pd.DataFrame -> convert to series
        return detected.isin(values=[hashed_image]).sum()

    def sim_initialization(self, data, x_col, y_col, detected=0.2):
        """

        :param pd.DataFrame data:
        :param str x_col:
        :param str y_col:
        :param str destination:
        :param float detected:
        """
        hateful = data.loc[data[y_col] == 1, :]
        hateful = hateful.sample(frac=1)  # shuffle
        detected_hateful = hateful.head(int(len(hateful) * detected))[x_col]
        initially_detected = detected_hateful.apply(func=self.img_path_to_hash)
        initially_detected.name = "detected"
        pd.DataFrame(initially_detected).to_csv(self.detected_path, index=False)


memes_path = "../../data/hateful_memes_data/"
detected_path = "../../data/exact_matching/detected.csv"
data = pd.read_json(memes_path + "train.jsonl", lines=True).loc[:, ["img", "label"]]
hashing_matcher = HashingMatcher(data_path=memes_path,
                                 detected_path=detected_path,
                                 hash_func=hashlib.sha224)
'''
# initialize the csv file having a number of detected hateful images
hashing_matcher.sim_initialization(data=data,
                                   x_col="img",
                                   y_col="label")
'''
#print(data["img"].apply(func=hashing_matcher.predict).sum())

transform_pipe = transforms.Compose([transforms.ToTensor()])
hash_func = hashlib.new("whirlpool")

a = torch.tensor([1, 2, 3])
a_bytes = bytes(str(a), "UTF-8")
hash_func.update(a_bytes)


b = torch.tensor([1, 2, 3])
b_bytes = bytes(str(b), "UTF-8")
hash_func.update(b_bytes)


print(hashlib.sha224(a_bytes).hexdigest() == hashlib.sha224(b_bytes).hexdigest())

