import librosa
import numpy as np
import soundfile as sf
import os

INPUT_DIR = "preprocessing/1_silence_trimmed"
OUTPUT_DIR = "preprocessing/2_fixed_length"

SR = 22050
DURATION = 30  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

target_len = SR * DURATION

for file in os.listdir(INPUT_DIR):
    if file.endswith(".wav"):
        input_path = os.path.join(INPUT_DIR, file)

        # Load audio
        y, sr = librosa.load(input_path, sr=SR)

        current_len = len(y)

        # Case 1: Longer than 30 sec → center crop
        if current_len > target_len:
            start = (current_len - target_len) // 2
            y = y[start:start + target_len]

        # Case 2: Shorter than 30 sec → pad at end
        elif current_len < target_len:
            y = np.pad(y, (0, target_len - current_len))

        # Case 3: Exactly 30 sec → keep as is

        output_path = os.path.join(OUTPUT_DIR, file)
        sf.write(output_path, y, SR)

print("Step 2: Fixed length processing completed successfully.")