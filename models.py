import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import *


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, nb_classes=NB_CLASS, num_layers=2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, nb_classes)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size)).to(device)
        c0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size)).to(device)

        return (h0, c0)

    def forward(self, x):
        # hidden = self.init_hidden(x.shape[0])
        x = self.lstm(x)[0]
        x = self.out(x)
        x = x[:, -1, :]

        return x


class FeatExtractCNN(nn.Module):
    def __init__(self):
        super(FeatExtractCNN, self).__init__()
        self.fe = models.densenet121(pretrained=True)
        self.fe.classifier = nn.Identity()

    def forward(self, X):
        return self.fe(X)


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.BatchNorm2d(),
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d(),
        )


class SalakhNet(nn.Module):
    def __init__(self, nb_class, input_size=FEAT_NUM):
        super(SalakhNet, self).__init__()
        # self.fc1 = nn.Linear(side * side, 500)
        # self.fc2 = nn.Linear(500, 500)
        # self.fc3 = nn.Linear(500, 2000)
        # self.fc4 = nn.Linear(2000, nb_class)
        #
        self.net = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, nb_class),
        )

    def forward(self, img):
        return self.net(img)


if __name__ == "__main__":
    model = BiLSTM(512)

    batch_size = 256
    inp = torch.rand(batch_size, SEQ_LENGTH, FEAT_NUM)

    out = model(inp)

    print(out.shape)
