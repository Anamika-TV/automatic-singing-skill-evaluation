import numpy as np
import librosa
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("singing_skill_model.h5")

SR = 22050
DURATION = 30
TARGET_LEN = SR * DURATION

def preprocess_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=SR)

    # Fixed length (center crop or pad)
    if len(y) > TARGET_LEN:
        start = (len(y) - TARGET_LEN) // 2
        y = y[start:start + TARGET_LEN]
    elif len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Add channel dimension
    mel_db = mel_db[..., np.newaxis]

    # Add batch dimension
    mel_db = np.expand_dims(mel_db, axis=0)

    return mel_db

# ---- Predict ----
file_path = "ClipL_kannodu.mp3"

processed_input = preprocess_audio(file_path)

prediction = model.predict(processed_input)

print("Predicted Singing Score:", float(prediction[0][0]))