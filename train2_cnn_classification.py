import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1️⃣ Load Dataset (classification labels)
# -----------------------------
FEATURE_DIR = "features/mel_spectrograms"
LABEL_FILE = "dataset/labels_classification.csv"   # file with filename,class (0,1,2)

labels_df = pd.read_csv(LABEL_FILE, names=["filename", "class"])

X = []
y = []
filenames = []

for index, row in labels_df.iterrows():
    filename_npy = row["filename"].replace(".wav", ".npy")
    class_label = int(row["class"])

    feature_path = os.path.join(FEATURE_DIR, filename_npy)

    if os.path.exists(feature_path):
        mel = np.load(feature_path)
        # Normalize each spectrogram
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)
        mel = mel[..., np.newaxis]          # add channel dimension

        X.append(mel)
        y.append(class_label)
        filenames.append(row["filename"])

X = np.array(X)
y = np.array(y)
filenames = np.array(filenames)

print("Dataset shape:", X.shape, y.shape)

# -----------------------------
# 2️⃣ Train / Validation / Test Split
# -----------------------------
X_train, X_temp, y_train, y_temp, fn_train, fn_temp = train_test_split(
    X, y, filenames, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test, fn_val, fn_test = train_test_split(
    X_temp, y_temp, fn_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\n========== DATA SPLIT ==========")
print(f"Train: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# -----------------------------
# 3️⃣ Build CNN Classification Model
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

    # Global average pooling
    layers.GlobalAveragePooling2D(),

    # Dense head
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(3, activation='softmax')   # 3 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 4️⃣ Class Weights (to handle imbalance)
# -----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights:", class_weight_dict)

# -----------------------------
# 5️⃣ Train Model
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
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 6️⃣ Evaluate on test set
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# -----------------------------
# 7️⃣ Predictions for all splits (print like regression)
# -----------------------------
def print_predictions(split_name, X_data, y_data, fn_data):
    print(f"\n========== {split_name} PREDICTIONS ==========")
    pred_probs = model.predict(X_data, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)
    for i in range(min(len(X_data), 10)):          # show first 10 only
        correct = "✓" if pred_classes[i] == y_data[i] else "✗"
        print(f"Song: {fn_data[i]}")
        print(f"Actual: {y_data[i]}  Predicted: {pred_classes[i]}  {correct}  Confidence: {np.max(pred_probs[i]):.2f}")
        print("-" * 40)
    if len(X_data) > 10:
        print(f"... and {len(X_data)-10} more")

print_predictions("TRAIN", X_train, y_train, fn_train)
print_predictions("VALIDATION", X_val, y_val, fn_val)
print_predictions("TEST", X_test, y_test, fn_test)

# -----------------------------
# 8️⃣ Overall accuracy on full dataset
# -----------------------------
full_pred_probs = model.predict(X, verbose=0)
full_pred_classes = np.argmax(full_pred_probs, axis=1)
overall_acc = accuracy_score(y, full_pred_classes)
print(f"\nOverall accuracy on full dataset ({len(X)} samples): {overall_acc:.4f}")

# -----------------------------
# 9️⃣ Save Model (optional)
# -----------------------------
model.save("2_cnn_classification_model.keras")
print("\nModel saved as 2_cnn_classification_model.keras")