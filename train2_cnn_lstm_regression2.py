import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
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
        
        # ---- IMPROVEMENT: Normalize each spectrogram ----
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)
        mel = mel[..., np.newaxis]          # add channel dimension

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
print(f"Train: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# ---- Compute mean of training labels for later comparison ----
train_mean = np.mean(y_train)
print(f"\nMean of training labels: {train_mean:.2f}")

# -----------------------------
# 3️⃣ Build CNN‑LSTM Model (with LSTM)
# -----------------------------
input_shape = (128, 1292, 1)

model = models.Sequential([
    layers.Input(shape=input_shape),

    # Conv block 1
    layers.Conv2D(8, (3,3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Conv block 2
    layers.Conv2D(16, (3,3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Conv block 3
    layers.Conv2D(16, (3,3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # At this point, the shape is (batch, 16, 161, 16)
    # Permute to bring the time dimension (width) to the front: (batch, 161, 16, 16)
    layers.Permute((2, 1, 3)),

    # Reshape to (batch, 161, 16*16) = (batch, 161, 256)
    layers.Reshape((161, 256)),

    # LSTM layer (return_sequences=False gives a single output per sample)
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.5),

    # Dense head
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(1)  # linear activation for regression
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
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 5️⃣ Evaluate Model
# -----------------------------
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print("\nTest MSE:", test_loss)
print("Test MAE:", test_mae)

# -----------------------------
# 6️⃣ Predictions & Check for Mean-Predicting Behaviour
# -----------------------------
def print_predictions(split_name, X_data, y_data, fn_data):
    print(f"\n========== {split_name} PREDICTIONS ==========")
    preds = model.predict(X_data, verbose=0)
    for i in range(min(len(X_data), 10)):          # show first 10 only
        print(f"Song: {fn_data[i]}")
        print(f"Actual: {y_data[i]:.2f}  Predicted: {float(preds[i][0]):.2f}  Error: {abs(preds[i][0]-y_data[i]):.2f}")
        print("-" * 40)
    if len(X_data) > 10:
        print(f"... and {len(X_data)-10} more")

print_predictions("TRAIN", X_train, y_train, fn_train)
print_predictions("VALIDATION", X_val, y_val, fn_val)
print_predictions("TEST", X_test, y_test, fn_test)

# ---- Check if model is just predicting the mean ----
all_preds = model.predict(X, verbose=0).flatten()
pred_mean = np.mean(all_preds)
pred_std = np.std(all_preds)
true_std = np.std(y)
r2 = r2_score(y, all_preds)

print("\n========== MODEL BEHAVIOR CHECK ==========")
print(f"Mean of all predictions: {pred_mean:.2f}  (training label mean: {train_mean:.2f})")
print(f"Standard deviation of predictions: {pred_std:.2f}  (true std: {true_std:.2f})")
print(f"R² score on entire dataset: {r2:.3f}")

if pred_std < 0.1 * true_std:
    print("⚠️  The model is predicting almost constant values (likely just the mean).")
elif r2 < 0.1:
    print("⚠️  Very low R² – model explains almost no variance.")
else:
    print("✅ The model is capturing some variance beyond the mean.")

# -----------------------------
# 7️⃣ Save Model (optional)
# -----------------------------
model.save("2_cnn_lstm_regression_model.keras")
print("\nModel saved as 2_cnn_lstm_regression_model.keras")