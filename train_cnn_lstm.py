import os
import numpy as np
from utils.features import extract_mfcc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

TIME_STEPS = 200
N_MFCC = 40

X = []
y = []

def fix_shape(mfcc):
    if mfcc.shape[1] < TIME_STEPS:
        mfcc = np.pad(mfcc, ((0,0),(0, TIME_STEPS - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :TIME_STEPS]
    mfcc = mfcc.T
    return mfcc

for file in os.listdir("data/human"):
    if not file.lower().endswith((".wav", ".mp3")):
        continue
    mfcc = extract_mfcc(os.path.join("data/human", file))
    if mfcc is None:
        continue
    mfcc = fix_shape(mfcc)
    X.append(mfcc)
    y.append(0)

for file in os.listdir("data/ai"):
    if not file.lower().endswith((".wav", ".mp3")):
        continue
    mfcc = extract_mfcc(os.path.join("data/ai", file))
    if mfcc is None:
        continue
    mfcc = fix_shape(mfcc)
    X.append(mfcc)
    y.append(1)

X = np.array(X)
y = np.array(y)

model = Sequential([
    Conv1D(64, 3, activation="relu", input_shape=(TIME_STEPS, N_MFCC)),
    MaxPooling1D(2),
    Conv1D(128, 3, activation="relu"),
    MaxPooling1D(2),
    LSTM(128),
    Dropout(0.4),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=15, batch_size=8)

os.makedirs("models", exist_ok=True)
model.save("models/voice_cnn_lstm.h5")
