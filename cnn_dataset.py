import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import *

import torchvision.models as models


class CNNDataset(Dataset):
    def __init__(self, split):
        between = SPLIT_METHOD + "_CNN"

        self.X = np.load(os.path.join(VARS_DIR, "X_" + between + "_" + split + ".npy"))
        self.Y = np.load(os.path.join(VARS_DIR, "Y_" + between + "_" + split + ".npy"))

    def __getitem__(self, idx):
        img = cv2.resize(cv2.imread(os.sep.join([IMAGES_DIR, self.X[idx]])))


        img_t = np.transpose(img, [2, 0, 1])

        label = self.Y[idx]

        return (img_t, label)

    def __len__(self):
        return self.df.shape[0]


def image_statistics():
    df = pd.read_csv(os.sep.join([CSV_DIR, "28_with_filenames.csv"]))

    shape = np.zeros(3)
    for i in range(df.shape[0]):
        row = df.iloc[i]
        img_path = os.sep.join([IMAGES_DIR, row.filename])
        img = cv2.imread(img_path)
        shape += img.shape

    shape = shape / df.shape[0]

    print(shape)


if __name__ == "__main__":
    image_statistics()
    # dataset = CNNDataset("train")
    #
    # for i in range(5):
    #     img_t, label = dataset[i]
    #     print(img_t.shape)
    #
    # model = models.densenet121(pretrained=True).features
    #
    # print(model)
    # for i in range(len(dataset)):
    #     img_t, label = dataset[i]
    #     img_t = img_t.unsqueeze(0)
    #
    #     out = model(img_t)
    #
    #
    #     break
