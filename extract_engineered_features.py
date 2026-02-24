import librosa
import numpy as np
import pandas as pd
import os

AUDIO_DIR = "preprocessing/3_rms_normalized"
LABEL_FILE = "dataset/labels_classification.csv"

OUTPUT_FEATURE_FILE = "features/engineered_features.npy"
OUTPUT_LABEL_FILE = "features/engineered_labels.npy"

labels_df = pd.read_csv(LABEL_FILE, names=["filename", "class"])

X = []
y = []

for index, row in labels_df.iterrows():
    filename = row["filename"]
    class_label = int(row["class"])

    path = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(path):
        continue

    y_audio, sr = librosa.load(path, sr=22050)

    # 1️⃣ MFCC
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # 2️⃣ Pitch (F0)
    pitches, magnitudes = librosa.piptrack(y=y_audio, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch_values) > 0:
        pitch_variance = np.var(pitch_values)
    else:
        pitch_variance = 0

    # 3️⃣ Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y_audio, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    # 4️⃣ Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y_audio)
    zcr_mean = np.mean(zcr)

    # 5️⃣ Harmonic Ratio (approx)
    harmonic, percussive = librosa.effects.hpss(y_audio)
    harmonic_energy = np.sum(np.abs(harmonic))
    percussive_energy = np.sum(np.abs(percussive)) + 1e-6
    harmonic_ratio = harmonic_energy / percussive_energy

    # Combine features
    features = np.hstack([
        mfcc_mean,
        pitch_variance,
        spectral_centroid_mean,
        zcr_mean,
        harmonic_ratio
    ])

    X.append(features)
    y.append(class_label)

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

np.save(OUTPUT_FEATURE_FILE, X)
np.save(OUTPUT_LABEL_FILE, y)

print("Engineered features saved successfully.")