import hashlib
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ClusterMatcher:
    def __init__(self, data_path, detected_path):
        self.data_path = data_path
        self.detected_path = detected_path

    def read_detected(self):
        return pd.read_csv(self.detected_path)

    def predict_cluster(self, img_path):
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
        self.hash_func.update(img_bytes)
        return self.hash_func.hexdigest()

    def predict(self, hashed_image):
        detected = pd.read_csv(self.detected_path)["detected"]  # stored as pd.DataFrame -> convert to series
        return detected.isin(values=[hashed_image]).sum()

    def sim_initialization(self, data, x_col, y_col, detected=0.2):
        """

        :param pd.DataFrame data:
        :param str x_col:
        :param str y_col:
        :param float detected:
        """
        hateful = data.loc[data[y_col] == 1, :]
        hateful = hateful.sample(frac=1)  # shuffle
        detected_hateful = hateful.head(int(len(hateful) * detected))[x_col]
        initially_detected = detected_hateful.apply(func=self.img_to_features)
        pd.DataFrame(np.stack(initially_detected.values)).to_csv(self.detected_path, index=False)

    def img_to_features(self, img_path):
        path = self.data_path + img_path
        transform_pipe = transforms.Compose([transforms.Resize([2, 2]), transforms.ToTensor()])

        image = Image.open(path, formats=["PNG"])
        tensor = transform_pipe(image)
        if tensor.size(0) == 4:  # very few images have one more channel, change to RGB format
            image = Image.open(path, formats=["PNG"]).convert("RGB")
            tensor = transform_pipe(image)
        return tensor.flatten().numpy()  # size 3 * 2 * 2

    def cluster(self):
        def perform_clustering(n_clusters, data):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
            return {"error": kmeans.inertia_, "n_clusters": n_clusters}
        errors = []
        n_clusters = []
        detected = self.read_detected()
        print(len(detected))

        for n in np.arange(start=int(len(detected)/11), stop=int(len(detected)), step=int(len(detected)/11)):
            results = perform_clustering(n_clusters=n, data=detected)
            error, n_cluster = results["error"], results["n_clusters"]
            errors.append(error)
            n_clusters.append(n_cluster)
        fig, ax = plt.subplots()
        ax.set_title("Elbow Criterion")
        ax.set_xlabel("n_clusters")
        ax.set_ylabel("error")
        plt.plot(n_clusters, errors)
        plt.show()


memes_path = "../../data/hateful_memes_data/"
detected_path = "../../data/exact_matching/detected.csv"
data = pd.read_json(memes_path + "train.jsonl", lines=True).loc[:, ["img", "label"]]
cluster_matcher = ClusterMatcher(data_path=memes_path,
                                 detected_path=detected_path)
'''
# initialize the csv file having a number of detected hateful images
cluster_matcher.sim_initialization(data=data,
                                   x_col="img",
                                   y_col="label")
'''
cluster_matcher.cluster()
#print(data["img"].apply(func=hashing_matcher.predict).sum())
