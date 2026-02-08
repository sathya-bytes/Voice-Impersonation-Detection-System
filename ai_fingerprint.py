import librosa, numpy as np, os

def fingerprint(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(d1, axis=1),
        np.mean(d2, axis=1)
    ])

fps = []

for file in os.listdir("data/Ai"):
    if not file.lower().endswith((".wav", ".mp3")):
        continue
    y, sr = librosa.load(os.path.join("data/Ai", file), sr=16000)
    fps.append(fingerprint(y, sr))

fp = np.mean(fps, axis=0)

os.makedirs("models", exist_ok=True)
np.save("models/ai_reference_fp.npy", fp)

print("AI reference fingerprint created:", fp.shape)
