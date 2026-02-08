import librosa
import numpy as np

def extract_mfcc(audio_path, max_len=200):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.T

    if mfcc.shape[0] < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:max_len, :]

    return mfcc
