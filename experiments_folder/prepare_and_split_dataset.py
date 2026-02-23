import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

FEATURE_DIR = "features/mel_spectrograms"
LABEL_FILE = "dataset/labels.csv"

labels_df = pd.read_csv(LABEL_FILE, names=["filename", "score"])

X = []
y = []
filenames = []

for index, row in labels_df.iterrows():
    filename_npy = row["filename"].replace(".wav", ".npy")
    score = row["score"]

    feature_path = os.path.join(FEATURE_DIR, filename_npy)

    if os.path.exists(feature_path):
        mel = np.load(feature_path)
        mel = mel[..., np.newaxis]

        X.append(mel)
        y.append(score)
        filenames.append(row["filename"])

X = np.array(X)
y = np.array(y)
filenames = np.array(filenames)

print("Full Dataset:", X.shape, y.shape)

# Split including filenames
X_train, X_temp, y_train, y_temp, fn_train, fn_temp = train_test_split(
    X, y, filenames, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test, fn_val, fn_test = train_test_split(
    X_temp, y_temp, fn_temp, test_size=0.50, random_state=42
)

print("\nTrain Songs:", fn_train)
print("\nValidation Songs:", fn_val)
print("\nTest Songs:", fn_test)