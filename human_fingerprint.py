import librosa, numpy as np, os

def fingerprint(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    fp = np.concatenate([mfcc.mean(1), d1.mean(1), d2.mean(1)])
    return fp / np.linalg.norm(fp)

fps = []

for file in os.listdir("data/human"):
    if file.endswith((".wav",".mp3")):
        y, sr = librosa.load(f"data/human/{file}", sr=16000)
        fps.append(fingerprint(y, sr))

human_fp = np.mean(fps, axis=0)

os.makedirs("models", exist_ok=True)
np.save("models/human_reference_fp.npy", human_fp)

print("Human reference fingerprint created:", human_fp.shape)
