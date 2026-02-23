import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
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

print("Dataset shape:", X.shape, y.shape)

# -----------------------------
# 2️⃣ Train / Validation / Test Split
# -----------------------------
X_train, X_temp, y_train, y_temp, fn_train, fn_temp = train_test_split(
    X, y, filenames, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test, fn_val, fn_test = train_test_split(
    X_temp, y_temp, fn_temp, test_size=0.50, random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Train Songs:")
for name in fn_train:
    print(name)

print("\nValidation Songs:")
for name in fn_val:
    print(name)

print("\nTest Songs:")
for name in fn_test:
    print(name)

print("\nShapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# -----------------------------
# 3️⃣ Build CNN-LSTM Model
# -----------------------------
input_shape = (128, 1292, 1)

model = models.Sequential([
    layers.Input(shape=input_shape),

    layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),

    layers.Dropout(0.3),

    layers.Reshape((32, 323 * 32)),

    layers.LSTM(64),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# -----------------------------
# 4️⃣ Train Model
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    callbacks=[early_stop]
)

# -----------------------------
# 5️⃣ Evaluate Model
# -----------------------------
test_loss, test_mae = model.evaluate(X_test, y_test)

print("\nTest MSE:", test_loss)
print("Test MAE:", test_mae)

# -----------------------------
# 6️⃣ Predictions for ALL Splits
# -----------------------------
def print_predictions(split_name, X_data, y_data, fn_data):
    print(f"\n========== {split_name} PREDICTIONS ==========")
    preds = model.predict(X_data)
    for i in range(len(X_data)):
        print(f"Song: {fn_data[i]}")
        print(f"Actual: {y_data[i]}")
        print(f"Predicted: {float(preds[i][0]):.2f}")
        print(f"Absolute Error: {abs(preds[i][0] - y_data[i]):.2f}")
        print("-" * 40)

print_predictions("TRAIN", X_train, y_train, fn_train)
print_predictions("VALIDATION", X_val, y_val, fn_val)
print_predictions("TEST", X_test, y_test, fn_test)

# -----------------------------
# 7️⃣ Save Model
# -----------------------------
model.save("singing_skill_model.h5")
print("\nModel saved as singing_skill_model.h5")