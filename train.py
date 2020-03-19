import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from models import BiLSTM, SalakhNet
from data_processing import load_data, load_cnn_data, generateXY, normalize
from config import *


def train(model, X_tr, Y_tr, X_val, Y_val, X_test, Y_test, n_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        X_val = torch.Tensor(X_val).to(device)
        X_test = torch.Tensor(X_test).to(device)
        preds = model(X_val).log_softmax(dim=1).cpu().numpy().argmax(axis=1)
        best_val_acc = np.sum(preds == Y_val) / len(preds) * 100

        preds = model(X_test).log_softmax(dim=1).cpu().numpy().argmax(axis=1)
        test_acc = np.sum(preds == Y_test) / len(preds) * 100

        print("BEST VAL ACC: %.2f" % best_val_acc, "TEST ACC %.2f" % test_acc)

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        n_batches = math.ceil(len(Y_tr) / BATCH_SIZE)
        for batch_idx in range(n_batches):
            optimizer.zero_grad()

            s = batch_idx * BATCH_SIZE
            e = min(len(Y_tr), (batch_idx + 1) * BATCH_SIZE)
            X_batch, Y_batch = torch.Tensor(X_tr[s:e]).to(device), torch.LongTensor(Y_tr[s:e]).to(device)

            preds = model(X_batch)
            loss = loss_fn(preds, Y_batch)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        # print("Train Loss:", np.mean(losses))

        with torch.no_grad():
            model.eval()
            preds = model(X_val).log_softmax(dim=1).cpu().numpy().argmax(axis=1)
            val_acc = np.sum(preds == Y_val) / len(preds) * 100
            print("Val ACC: %.2f" % val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                preds = model(X_test).log_softmax(dim=1).cpu().numpy().argmax(axis=1)
                test_acc = np.sum(preds == Y_test) / len(preds) * 100
                torch.save(model.state_dict(), MODEL_PATH)
                print("Model Saved, Test ACC: %.2f " % test_acc)


if __name__ == "__main__":
    if use_CNN:
        X_tr, Y_tr, X_val, Y_val, X_test, Y_test = load_cnn_data()
    else:
        try:
            X_tr, Y_tr, X_val, Y_val, X_test, Y_test = load_data()
        except:
            generateXY()
            X_tr, Y_tr, X_val, Y_val, X_test, Y_test = load_data()

        X_tr, X_val, X_test = normalize([X_tr, X_val, X_test])
    device = torch.device("cuda:0")
    if SEQ_LENGTH > 1:
        model = BiLSTM(256).to(device)
    else:
        model = SalakhNet(nb_class=NB_CLASS).to(device)

    load = True
    if load and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Model Loaded")

    train(model, X_tr, Y_tr, X_val, Y_val, X_test, Y_test, n_epochs=100)
