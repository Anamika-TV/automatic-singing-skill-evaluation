import librosa
import numpy as np
import os

INPUT_DIR = "preprocessing/3_rms_normalized"
OUTPUT_DIR = "features/mel_spectrograms"

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".wav"):
        input_path = os.path.join(INPUT_DIR, file)

        # Load audio
        y, sr = librosa.load(input_path, sr=SR)

        # Compute Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        # Convert to Log scale
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Save as numpy array
        output_path = os.path.join(OUTPUT_DIR, file.replace(".wav", ".npy"))
        np.save(output_path, mel_db)

print("Step 4: Log-Mel spectrogram extraction completed.")