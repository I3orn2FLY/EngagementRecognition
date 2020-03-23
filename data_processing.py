import pandas as pd
import numpy as np
import glob
import torch
import time
from models import FeatExtractCNN
from PIL import Image
from torchvision import transforms
from config import *

np.random.seed(0)


def save_XY(X, Y, split, shuffle=True):
    if shuffle:
        idxs = list(range(len(Y)))
        np.random.shuffle(idxs)
        X, Y = X[idxs], Y[idxs]

    between_name = get_between_name(SPLIT_METHOD, SEQ_LENGTH, FEAT_MODEL, FEAT_NUM)
    np.save(os.path.join(VARS_DIR, "X_" + between_name + "_" + split), X)
    np.save(os.path.join(VARS_DIR, "Y_" + between_name + "_" + split), Y)

    print(split, "shapes:", X.shape, Y.shape, "between_name:", between_name)
    if split == "train":
        X = X.reshape(-1, FEAT_NUM)
        mean_x = np.mean(X, axis=0)
        std_x = np.std(X, axis=0)
        np.save(os.path.join(VARS_DIR, "X_" + between_name + "_mean"), mean_x)
        np.save(os.path.join(VARS_DIR, "X_" + between_name + "_std"), std_x)


def load_data():
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

    return X_tr, Y_tr, X_val, Y_val, X_test, Y_test


def show(start_time, cur_idx, step, L):
    cur_idx += 1

    time_left = int((time.time() - start_time) * 1.0 / cur_idx * (L - cur_idx))

    hours = time_left // 3600

    minutes = time_left % 3600 // 60

    seconds = time_left % 60

    print("\rProgress: %.2f" % (cur_idx * 100 / L) + "% " \
          + str(hours) + " hours " \
          + str(minutes) + " minutes " \
          + str(seconds) + " seconds left",
          end=" ")





def get_columns():
    cols = ["frameID"]

    if FEAT_MODEL == "pose":
        with_body = False
        with_face = False
        with_hands = False
        if FEAT_NUM == 411:
            with_body = True
            with_face = True
            with_hands = True
        elif FEAT_NUM == 201:
            with_body = True
            with_hands = True
        elif FEAT_NUM == 210:
            with_face = True

        if with_body:
            for i in range(25):
                cols.append("poseX" + str(i))
                cols.append("poseY" + str(i))
                cols.append("poseC" + str(i))

        if with_hands:
            for i in range(21):
                cols.append("handLeftX" + str(i))
                cols.append("handLeftY" + str(i))
                cols.append("handLeftC" + str(i))

            for i in range(21):
                cols.append("handRightX" + str(i))
                cols.append("handRightY" + str(i))
                cols.append("handRightC" + str(i))

        if with_face:
            for i in range(70):
                cols.append("faceX" + str(i))
                cols.append("faceY" + str(i))
                cols.append("faceC" + str(i))

    elif FEAT_MODEL == "densenet121":
        cols.append("filename")

    cols.append("engagement")

    return cols


def get_XY_by_df(df_s):
    X = []
    Y = []

    df_s.sort_values(by=['frameID'], inplace=True)
    video = df_s.values[:, 1:]
    n = video.shape[0] // SEQ_LENGTH

    for i in range(n):
        if SEQ_LENGTH > 1:
            seq = video[i * SEQ_LENGTH: (i + 1) * SEQ_LENGTH]

            x = seq[:, :-1]

            if FEAT_MODEL == "densenet121":
                poses = []
                for img_path in x:
                    pose_path = os.path.join(POSE_DIR, os.path.splitext(img_path[0])[0] + ".npy")
                    pose = np.load(pose_path)
                    poses.append(pose)

                x = np.stack(poses)

            targets = seq[:, -1].astype(np.int8)
            (values, counts) = np.unique(targets, return_counts=True)

            y = values[np.argmax(counts)]
        else:
            x = video[i, :-1][0]
            if FEAT_MODEL == "densenet121":
                pose_path = os.path.join(POSE_DIR, os.path.splitext(x)[0] + ".npy")
                x = np.load(pose_path)

            y = video[i, -1]

        X.append(x)
        Y.append(y)

    return X, Y


def generate_split(df, cols, session_ids, child_ids):
    if SPLIT_METHOD == "RANDOM":
        df_s = df[cols]
        X, Y = get_XY_by_df(df_s)
    else:
        X = []
        Y = []
        for ch_id in child_ids:
            for sess_id in session_ids:
                df_s = df.loc[np.logical_and(df["childID"] == ch_id, df["sessionID"] == sess_id)][cols]
                x, y = get_XY_by_df(df_s)
                X += x
                Y += y

    X = np.stack(X)
    Y = np.stack(Y).astype(np.int8)

    return X, Y


def generateXY(data_path=CSV_FILE):
    print("Generating Data.")
    print("Split method:", SPLIT_METHOD)
    print("Sequence Length:", SEQ_LENGTH)
    print("Model path/name:", get_model_path(SPLIT_METHOD, SEQ_LENGTH, FEAT_MODEL, FEAT_NUM))

    df = pd.read_csv(data_path)
    child_ids = df.childID.unique()
    session_ids = df.sessionID.unique()

    cols = get_columns()

    df = df[["childID", "sessionID"] + cols]

    if FEAT_MODEL == "densenet121":
        model = FeatExtractCNN().to(DEVICE)
        model.eval()

    if os.path.exists(os.sep.join([VARS_DIR, "child_ids.npy"])):
        child_ids = np.load(os.sep.join([VARS_DIR, "child_ids.npy"]))
    else:
        np.random.shuffle(child_ids)
        np.save(os.sep.join([VARS_DIR, "child_ids"]), child_ids)

    if os.path.exists(os.sep.join([VARS_DIR, "session_ids.npy"])):
        session_ids = np.load(os.sep.join([VARS_DIR, "session_ids.npy"]))
    else:
        np.random.shuffle(session_ids)
        np.save(os.sep.join([VARS_DIR, "session_ids"]), session_ids)

    if SPLIT_METHOD == "SESSION":
        session_ids_tr = session_ids[:int(0.8 * len(session_ids))]
        X_tr, Y_tr = generate_split(df, cols, session_ids_tr, child_ids)
        save_XY(X_tr, Y_tr, split="train")

        session_ids_val = session_ids[int(0.8 * len(session_ids)):int(0.9 * len(session_ids))]
        X_val, Y_val = generate_split(df, cols, session_ids_val, child_ids)
        save_XY(X_val, Y_val, split="val")

        session_ids_test = session_ids[int(0.9 * len(session_ids)):]
        X_test, Y_test = generate_split(df, cols, session_ids_test, child_ids)
        save_XY(X_test, Y_test, split="test")

    elif SPLIT_METHOD == "CHILD":
        child_ids_tr = child_ids[:int(0.8 * len(child_ids))]
        X_tr, Y_tr = generate_split(df, cols, session_ids, child_ids_tr)

        child_ids_val = child_ids[int(0.8 * len(child_ids)):int(0.9 * len(child_ids))]
        X_val, Y_val = generate_split(df, cols, session_ids, child_ids_val)

        child_ids_test = child_ids[int(0.9 * len(child_ids)):]
        X_test, Y_test = generate_split(df, cols, session_ids, child_ids_test)

        save_XY(X_tr, Y_tr, split="train")
        save_XY(X_val, Y_val, split="val")
        save_XY(X_test, Y_test, split="test")
    else:  # RANDOM SPLIT
        X, Y = generate_split(df, cols, session_ids, child_ids)
        idxs = list(range(len(Y)))
        np.random.shuffle(idxs)
        X, Y = X[idxs], Y[idxs]

        X_tr, Y_tr = X[:int(0.8 * len(Y))], Y[:int(0.8 * len(Y))]
        X_val, Y_val = X[int(0.8 * len(Y)):int(0.9 * len(Y))], Y[int(0.8 * len(Y)):int(0.9 * len(Y))]
        X_test, Y_test = X[int(0.9 * len(Y)):], Y[int(0.9 * len(Y)):]

        save_XY(X_tr, Y_tr, split="train")
        save_XY(X_val, Y_val, split="val")
        save_XY(X_test, Y_test, split="test")


def add_filenames(in_csv_file, out_csv_file):
    df = pd.read_csv(os.sep.join([DATA_DIR, in_csv_file]))

    df.dropna(inplace=True)
    df.childID = df.childID.astype(np.int8)
    df.sessionID = df.sessionID.astype(np.int8)
    df.frameID = df.frameID.astype(np.int32)
    df.engagement = df.engagement.astype(np.int8)

    if "filename" not in df.columns:

        image_file_table = {}

        ls = glob.glob(os.sep.join([IMAGES_DIR, "**", "*.jpg"]), recursive=True)
        for image_file in ls:
            image_file = os.sep.join(image_file.split(os.sep)[-2:])
            folder, image_name = image_file.split(os.sep)
            child_id, session_id = folder.split("_")
            child_id = int(child_id[1:])
            session_id = int(session_id[1:])

            image_name = os.path.splitext(image_name)[0]
            frame_id = int(image_name.split("_")[4])

            key = (child_id, session_id, frame_id)

            image_file_table[key] = image_file

        filenames = []
        rms = 0
        for i in range(df.shape[0]):
            i = i - rms
            row = df.iloc[i]
            key = (int(row.childID), int(row.sessionID), int(row.frameID))
            if key not in image_file_table:
                df.drop(df.index[i], inplace=True)
                rms += 1
            else:
                filenames.append(image_file_table[key])

        df["filename"] = filenames

    print(df.head())
    df.to_csv(os.sep.join([DATA_DIR, out_csv_file]), index=None)


def generate_cnn_features(batch_size=CNN_BATCH_SIZE):
    print("Generating CNN Features")
    df = pd.read_csv(CSV_FILE)
    filenames = df.filename.values
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = FeatExtractCNN().to(DEVICE)
    model.eval()

    batches = []
    batch = []
    for i, filename in enumerate(filenames):
        img_path = os.path.join(IMAGES_DIR, filename)
        pose_path = os.path.join(POSE_DIR, os.path.splitext(filename)[0] + ".npy")

        if not os.path.exists(img_path) or os.path.exists(pose_path):
            continue

        pose_dir = os.path.split(pose_path)[0]
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir)

        batch.append((img_path, pose_path))
        if len(batch) == batch_size or i == len(filenames) - 1:
            batches.append(batch)
            batch = []

    with torch.no_grad():
        pp = ProgressPrinter(len(batches), 1)
        for idx, batch in enumerate(batches):
            if not batch: continue
            X = []
            for img_path, pose_path in batch:
                X.append(preprocess(Image.open(img_path)))

            X = torch.stack(X).to(DEVICE)
            out = model(X).cpu().numpy()

            for i, (img_path, pose_path) in enumerate(batch):
                feat = out[i]
                np.save(pose_path, feat)

            pp.show(idx)

        pp.end()


if __name__ == "__main__":
    # add_filenames("28_with_filenames.csv", "28_with_filenames.csv")
    # if FEAT_MODEL == "densenet121":
    #     generate_cnn_features()

    # for SPLIT_METHOD in ["SESSION", "RANDOM", "CHILD"]:
    #     for SEQ_LENGTH in [1, 5, 25, 50, 100]:
    #         try:
    #             load_data()
    #         except:
    #             generateXY()
    #
    #         print()


    generateXY()

