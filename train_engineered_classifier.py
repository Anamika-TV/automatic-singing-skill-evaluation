import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# 1️⃣ Load Features
# -------------------------
X = np.load("features/engineered_features.npy")
y = np.load("features/engineered_labels.npy")

print("Loaded feature shape:", X.shape)

# -------------------------
# 2️⃣ Normalize Features
# -------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------
# 3️⃣ Train / Val / Test Split
# -------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# -------------------------
# 4️⃣ Build Dense Model
# -------------------------
model = models.Sequential([
    layers.Input(shape=(17,)),

    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# 5️⃣ Train
# -------------------------
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

# -------------------------
# 6️⃣ Evaluate
# -------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_acc)


# -------------------------
# Add Evaluation Metrics >>>>>>>>>>>>>>>>>
# -------------------------

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Predict on test set
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -------------------------
# Classification Report
# -------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# -------------------------
# Plot Confusion Matrix (No seaborn)
# -------------------------
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()

classes = ["Bad (0)", "Intermediate (1)", "Good (2)"]
plt.xticks(np.arange(3), classes, rotation=45)
plt.yticks(np.arange(3), classes)

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center", color="black")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------
# finished >>>>>>>>>>>>>>>>>>>>>>
# -------------------------



# -------------------------
# 7️⃣ Predictions
# -------------------------
print("\nTest Predictions:\n")

predictions = model.predict(X_test)

for i in range(len(X_test)):
    predicted_class = np.argmax(predictions[i])

    print(f"Actual Class: {y_test[i]}")
    print(f"Predicted Class: {predicted_class}")
    print("-" * 30)

# Save model
model.save("engineered_feature_classifier.keras")
print("\nEngineered feature classifier saved.")