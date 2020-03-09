import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from config import *
from PIL import Image
import torchvision.models as models

class CNNDataset(Dataset):
    def __init__(self, input_csv):
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.df = pd.read_csv(input_csv)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_file = os.sep.join(
            [IMAGES_DIR, 'c' + str(int(row.childID)) + '_s' + str(int(row.sessionID)), str(int(row.frameID)) + ".jpg"])

        label = int(row.engagement)
        img = Image.open(img_file)

        img_t = self.preprocess(img)

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
    # dataset = CNNDataset(os.sep.join([CSV_DIR, "labels.csv"]))
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

