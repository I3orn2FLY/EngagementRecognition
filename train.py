import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from models import BiLSTM, SalakhNet
from config import *


def normalize(X):
    between_name = get_between_name(SPLIT_METHOD, SEQ_LENGTH, FEAT_MODEL, FEAT_NUM)
    mean_x = np.load(os.path.join(VARS_DIR, "X_" + between_name + "_mean.npy"))
    std_x = np.load(os.path.join(VARS_DIR, "X_" + between_name + "_std.npy"))

    if isinstance(X, list):
        X_norm = []
        for x in X:
            X_norm.append((x - mean_x) / std_x)

        return X_norm

    return (X - mean_x) / std_x


def train():
    print("Training the model.")
    print("Split method:", SPLIT_METHOD)
    print("Sequence Length:", SEQ_LENGTH)
    model_path = get_model_path(SPLIT_METHOD, SEQ_LENGTH, FEAT_MODEL, FEAT_NUM)
    print("Model path/name:", model_path)
    n_epochs = 200

    between_name = get_between_name(SPLIT_METHOD, SEQ_LENGTH, FEAT_MODEL, FEAT_NUM)
    X_tr = np.load(os.path.join(VARS_DIR, "X_" + between_name + "_train.npy"))
    Y_tr = np.load(os.path.join(VARS_DIR, "Y_" + between_name + "_train.npy"))
    X_val = np.load(os.path.join(VARS_DIR, "X_" + between_name + "_val.npy"))
    Y_val = np.load(os.path.join(VARS_DIR, "Y_" + between_name + "_val.npy"))
    X_test = np.load(os.path.join(VARS_DIR, "X_" + between_name + "_test.npy"))
    Y_test = np.load(os.path.join(VARS_DIR, "Y_" + between_name + "_test.npy"))

    print(X_tr.shape, Y_tr.shape)
    print(X_val.shape, Y_val.shape)
    print(X_test.shape, Y_test.shape)

    X_tr, X_val, X_test = normalize([X_tr, X_val, X_test])

    if SEQ_LENGTH > 1:
        model = BiLSTM(FEAT_NUM, 256, nb_classes=NB_CLASS).to(DEVICE)
    else:
        model = SalakhNet(input_size=FEAT_NUM, nb_class=NB_CLASS).to(DEVICE)

    load = True
    if load and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model Loaded")

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

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        n_batches = math.ceil(len(Y_tr) / get_batch_size(SEQ_LENGTH))
        for batch_idx in range(n_batches):
            optimizer.zero_grad()

            s = batch_idx * get_batch_size(SEQ_LENGTH)
            e = min(len(Y_tr), (batch_idx + 1) * get_batch_size(SEQ_LENGTH))
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
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                preds = model(X_test).log_softmax(dim=1).cpu().numpy().argmax(axis=1)
                test_acc = np.sum(preds == Y_test) / len(preds) * 100
                torch.save(model.state_dict(), model_path)

    print("Val ACC: %.2f" % best_val_acc, "Test Acc: %.2f" % test_acc)


if __name__ == "__main__":
    for SPLIT_METHOD in ["CHILD"]:
        for SEQ_LENGTH in [1, 5, 25, 50, 100]:
            train()
            print()
            print()
