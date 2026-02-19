import librosa
import numpy as np
import soundfile as sf
import os

INPUT_DIR = "preprocessing/2_fixed_length"
OUTPUT_DIR = "preprocessing/3_rms_normalized"

SR = 22050
TARGET_RMS = 0.1   # Safe RMS level

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".wav"):
        input_path = os.path.join(INPUT_DIR, file)

        y, sr = librosa.load(input_path, sr=SR)

        # Compute current RMS
        rms = np.sqrt(np.mean(y**2))

        # Avoid division by zero
        if rms > 0:
            y = y * (TARGET_RMS / rms)

        output_path = os.path.join(OUTPUT_DIR, file)
        sf.write(output_path, y, SR)

print("Step 3: RMS normalization completed successfully.")