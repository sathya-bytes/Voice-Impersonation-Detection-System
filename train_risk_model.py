import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier

def extract_behaviour(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except:
        return None

    rms = librosa.feature.rms(y=y)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return [
        float(np.mean(rms)),
        float(np.var(rms)),
        float(tempo.item())
    ]

X = []
y = []

for file in os.listdir("data/human"):
    if not file.lower().endswith((".wav", ".mp3")):
        continue
    features = extract_behaviour(os.path.join("data/human", file))
    if features is None:
        continue
    X.append(features)
    y.append(0)

for file in os.listdir("data/ai"):
    if not file.lower().endswith((".wav", ".mp3")):
        continue
    features = extract_behaviour(os.path.join("data/ai", file))
    if features is None:
        continue
    X.append(features)
    y.append(1)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/behaviour_rf.pkl")
