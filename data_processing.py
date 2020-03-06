import pandas as pd
import numpy as np
from config import *

np.random.seed(0)


def save_XY(X, Y, suffix, shuffle=True):
    if shuffle:
        idxs = list(range(len(Y)))
        np.random.shuffle(idxs)
        X, Y = X[idxs], Y[idxs]

    print(suffix, "shapes:", X.shape, Y.shape)
    suffix_ = SPLIT_METHOD + "_" + str(SEQ_LENGTH) + "_" + str(FEAT_NUM) + "_" + suffix
    np.save(os.path.join(VARS_DIR, "X_" + suffix_), X)
    np.save(os.path.join(VARS_DIR, "Y_" + suffix_), Y)

    if suffix == "train":
        X = X.reshape(-1, FEAT_NUM)
        mean_x = np.mean(X, axis=0)
        std_x = np.std(X, axis=0)
        np.save(os.path.join(VARS_DIR, "X_" + str(FEAT_NUM) + "_mean"), mean_x)
        np.save(os.path.join(VARS_DIR, "X_" + str(FEAT_NUM) + "_std"), std_x)


def load_data():
    between = SPLIT_METHOD + "_" + str(SEQ_LENGTH) + "_" + str(FEAT_NUM)
    X_tr = np.load(os.path.join(VARS_DIR, "X_" + between + "_train.npy"))
    y_tr = np.load(os.path.join(VARS_DIR, "Y_" + between + "_train.npy"))
    X_val = np.load(os.path.join(VARS_DIR, "X_" + between + "_val.npy"))
    y_val = np.load(os.path.join(VARS_DIR, "Y_" + between + "_val.npy"))
    X_test = np.load(os.path.join(VARS_DIR, "X_" + between + "_test.npy"))
    y_test = np.load(os.path.join(VARS_DIR, "Y_" + between + "_test.npy"))

    print(X_tr.shape, y_tr.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    return X_tr, y_tr, X_val, y_val, X_test, y_test


def normalize(X):
    mean_x = np.load(os.path.join(VARS_DIR, "X_" + str(FEAT_NUM) + "_mean.npy"))
    std_x = np.load(os.path.join(VARS_DIR, "X_" + str(FEAT_NUM) + "_std.npy"))

    if isinstance(X, list):
        X_norm = []
        for x in X:
            X_norm.append((x - mean_x) / std_x)

        return X_norm

    return (X - mean_x) / std_x


def get_columns():
    cols = ["frameID"]
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

    cols.append("engagement")

    return cols


def generate_split(df, cols, session_ids, child_ids):
    X = []
    Y = []
    for idx, ch_id in enumerate(child_ids):

        for sess_id in session_ids:
            df_s = df.loc[np.logical_and(df["childID"] == ch_id, df["sessionID"] == sess_id)][cols]
            df_s.sort_values(by=['frameID'], inplace=True)
            video = df_s.values[:, 1:]
            n = video.shape[0] // SEQ_LENGTH

            for i in range(n):
                if SEQ_LENGTH > 1:
                    seq = video[i * SEQ_LENGTH: (i + 1) * SEQ_LENGTH]
                    x = seq[:, :-1]
                    targets = seq[:, -1].astype(np.int8)
                    (values, counts) = np.unique(targets, return_counts=True)
                    y = values[np.argmax(counts)]
                else:
                    x = video[i, :-1]
                    y = video[i, -1]

                X.append(x)
                Y.append(y)

    X = np.array(X)
    Y = np.array(Y).astype(np.int8)

    return X, Y


def generateXY(data_path=CSV_FILE):
    df = pd.read_csv(data_path)
    child_ids = df.childID.unique()
    session_ids = df.sessionID.unique()

    cols = get_columns()

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
        save_XY(X_tr, Y_tr, suffix="train")

        session_ids_val = session_ids[int(0.8 * len(session_ids)):int(0.9 * len(session_ids))]
        X_val, Y_val = generate_split(df, cols, session_ids_val, child_ids)
        save_XY(X_val, Y_val, suffix="val")

        session_ids_test = session_ids[int(0.9 * len(session_ids)):]
        X_test, Y_test = generate_split(df, cols, session_ids_test, child_ids)
        save_XY(X_test, Y_test, suffix="test")

    elif SPLIT_METHOD == "CHILD":
        child_ids_tr = child_ids[:int(0.8 * len(child_ids))]
        X_tr, Y_tr = generate_split(df, cols, session_ids, child_ids_tr)

        child_ids_val = child_ids[int(0.8 * len(child_ids)):int(0.9 * len(child_ids))]
        X_val, Y_val = generate_split(df, cols, session_ids, child_ids_val)

        child_ids_test = child_ids[int(0.9 * len(child_ids)):]
        X_test, Y_test = generate_split(df, cols, session_ids, child_ids_test)

        save_XY(X_tr, Y_tr, suffix="train")
        save_XY(X_val, Y_val, suffix="val")
        save_XY(X_test, Y_test, suffix="test")
    else:  # RANDOM SPLIT
        X, Y = generate_split(df, cols, session_ids, child_ids)
        idxs = list(range(len(Y)))
        np.random.shuffle(idxs)
        X, Y = X[idxs], Y[idxs]

        X_tr, Y_tr = X[:int(0.8 * len(Y))], Y[:int(0.8 * len(Y))]
        X_val, Y_val = X[int(0.8 * len(Y)):int(0.9 * len(Y))], Y[int(0.8 * len(Y)):int(0.9 * len(Y))]
        X_test, Y_test = X[int(0.9 * len(Y)):], Y[int(0.9 * len(Y)):]

        save_XY(X_tr, Y_tr, suffix="train")
        save_XY(X_val, Y_val, suffix="val")
        save_XY(X_test, Y_test, suffix="test")


if __name__ == "__main__":
    generateXY()
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_data()

    # crit1 = "childID"
    # crit2 = "sessionID"
    # 
    # if SPLIT_METHOD == "SESSION":
    #     crit1, crit2 = crit2, crit1
    # 
    # crit_lists = {"childID": child_ids, "sessionID": session_ids}
    # 
    # for idx, val1 in enumerate(crit_lists[crit1]):
    #     if idx == int(0.8 * len(crit_lists[crit1])):
    #         save_XY(X, Y, suffix="train")
    # 
    #         X, Y = X_val, Y_val
    # 
    #     if idx == int(0.9 * len(crit_lists[crit1])):
    #         save_XY(X, Y, suffix="val")
    #         X, Y = X_test, Y_test
    # 
    #     for val2 in crit_lists[crit2]:
    #         df_s = df.loc[np.logical_and(df[crit1] == val1, df[crit2] == val2)][cols]
    #         df_s.sort_values(by=['frameID'], inplace=True)
    #         video = df_s.values[:, 1:]
    #         n = video.shape[0] // SEQ_LENGTH
    # 
    #         for i in range(n):
    #             if SEQ_LENGTH > 1:
    #                 seq = video[i * SEQ_LENGTH: (i + 1) * SEQ_LENGTH]
    #                 x = seq[:, :-1]
    #                 targets = seq[:, -1].astype(np.int8)
    #                 (values, counts) = np.unique(targets, return_counts=True)
    #                 y = values[np.argmax(counts)]
    #             else:
    #                 x = video[i, :-1]
    #                 y = video[i, -1]
    # 
    #             X.append(x)
    #             Y.append(y)
    # 
    # save_XY(X, Y, suffix="test")
