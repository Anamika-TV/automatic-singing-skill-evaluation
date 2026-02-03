import librosa
import numpy as np
import soundfile as sf
import os

INPUT_DIR = "audio"
OUTPUT_DIR = "audio_standardized"
DURATION = 40          # seconds
SR = 22050             # sampling rate

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".wav"):
        path = os.path.join(INPUT_DIR, file)

        # load audio
        audio, sr = librosa.load(path, sr=SR)

        target_len = DURATION * SR

        # trim or pad
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))

        # save
        sf.write(os.path.join(OUTPUT_DIR, file), audio, SR)
