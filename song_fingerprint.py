import librosa
import numpy as np
import os

y, sr = librosa.load("special_song.mp3", sr=16000)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
fp = np.mean(mfcc, axis=1)

meta = {
    "fp": fp,
    "duration": librosa.get_duration(y=y, sr=sr)
}

os.makedirs("models", exist_ok=True)
np.save("models/special_song_fp.npy", meta)

print("Special song fingerprint locked")
