import numpy as np
import matplotlib.pyplot as plt
import librosa.display

mel = np.load("features/mel_spectrograms/_bmNEUrlKCA.npy")

print(mel.shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel, sr=22050, hop_length=512,
                         x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Log-Mel Spectrogram")
plt.tight_layout()
plt.show()
#plt.savefig("mel_example.png")


