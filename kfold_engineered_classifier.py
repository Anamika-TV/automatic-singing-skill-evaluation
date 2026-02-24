import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# 1️⃣ Load Engineered Features
# -----------------------------
X = np.load("features/engineered_features.npy")
y = np.load("features/engineered_labels.npy")

print("Dataset shape:", X.shape)

# -----------------------------
# 2️⃣ Define K-Fold
# -----------------------------
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_accuracies = []
fold_number = 1

# -----------------------------
# 3️⃣ K-Fold Training Loop
# -----------------------------
for train_index, test_index in kf.split(X):

    print(f"\n========== Fold {fold_number} ==========")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 🔹 IMPORTANT: Scale inside each fold (no data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Build Fresh Model Each Fold
    # -----------------------------
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
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

    # -----------------------------
    # Train
    # -----------------------------
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        verbose=0
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    acc = accuracy_score(y_test, y_pred)

    print("Fold Accuracy:", round(acc, 4))

    fold_accuracies.append(acc)
    fold_number += 1

# -----------------------------
# 4️⃣ Final Result
# -----------------------------
print("\n==============================")
print("K-Fold Cross Validation Result")
print("==============================")
print("Fold Accuracies:", [round(a, 4) for a in fold_accuracies])
print("Mean Accuracy:", round(np.mean(fold_accuracies), 4))
print("Std Deviation:", round(np.std(fold_accuracies), 4))