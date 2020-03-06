import os

# SPLIT_METHOD = "CHILD"
# SPLIT_METHOD = "SESSION"
SPLIT_METHOD = "RANDOM"

# WITH FACE, BODY, HANDS (411) =>
# FEAT_NUM = 411

# WITH BODY AND HANDS (201) =>
# FEAT_NUM = (25 + 21 + 21) * 3

# WITH FACE ONLY (210) =>
FEAT_NUM = 70 * 3

SEQ_LENGTH = 5

NB_CLASS = 6
CSV_FILE = "/media/kenny/Extra/EngagementRecognition/Data/ASD_28_result.csv"
VARS_DIR = "/media/kenny/Extra/EngagementRecognition/VARS"

MODEL_PATH = os.sep.join(["weights", "BiLSTM_" + SPLIT_METHOD + "_" + str(SEQ_LENGTH) + "_" + str(FEAT_NUM) + ".pt"])

if SEQ_LENGTH == 1:
    MODEL_PATH = MODEL_PATH.replace("BiLSTM", "NN")

if SEQ_LENGTH == 1:
    BATCH_SIZE = 16384
elif SEQ_LENGTH == 5:
    BATCH_SIZE = 8192
elif SEQ_LENGTH == 25:
    BATCH_SIZE = 4096
elif SEQ_LENGTH == 50:
    BATCH_SIZE = 2048
else:
    BATCH_SIZE = 1024
