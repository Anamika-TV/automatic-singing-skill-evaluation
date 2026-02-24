import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1️⃣ Load Classification Dataset
# -----------------------------
FEATURE_DIR = "features/mel_spectrograms"
LABEL_FILE = "dataset/labels_classification.csv"

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
        mel = mel[..., np.newaxis]

        X.append(mel)
        y.append(class_label)
        filenames.append(row["filename"])

X = np.array(X)
y = np.array(y)
filenames = np.array(filenames)

print("Dataset shape:", X.shape, y.shape)

# -----------------------------
# 2️⃣ Split Dataset
# -----------------------------
X_train, X_temp, y_train, y_temp, fn_train, fn_temp = train_test_split(
    X, y, filenames, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test, fn_val, fn_test = train_test_split(
    X_temp, y_temp, fn_temp, test_size=0.50, random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# -----------------------------
# 3️⃣ Build CNN Classification Model
# -----------------------------
input_shape = (128, 1292, 1)

model = models.Sequential([
    layers.Input(shape=input_shape),

    layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),

    layers.Dropout(0.3),

    layers.GlobalAveragePooling2D(),  # << IMPORTANT CHANGE

    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Add Class Weights
# -----------------------------

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

# -----------------------------
# 4️⃣ Train
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
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# -----------------------------
# 5️⃣ Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_acc)

# -----------------------------
# 6️⃣ Per-song Predictions
# -----------------------------
print("\nTest Predictions:\n")

predictions = model.predict(X_test)

for i in range(len(X_test)):
    predicted_class = np.argmax(predictions[i])

    print(f"Song: {fn_test[i]}")
    print(f"Actual Class: {y_test[i]}")
    print(f"Predicted Class: {predicted_class}")
    print("-" * 40)

# Save model
model.save("singing_skill_classification_model.h5")
print("\nClassification model saved.")