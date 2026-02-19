import librosa
import soundfile as sf
import os

INPUT_DIR = "dataset/raw_audio"
OUTPUT_DIR = "preprocessing/1_silence_trimmed"

SR = 22050

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".wav"):
        path = os.path.join(INPUT_DIR, file)

        y, sr = librosa.load(path, sr=SR)

        # Trim silence (only start & end)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)

        out_path = os.path.join(OUTPUT_DIR, file)
        sf.write(out_path, y_trimmed, SR)

print("Silence trimming completed.")
