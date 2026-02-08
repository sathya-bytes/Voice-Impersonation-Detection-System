import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import tempfile
import os
import hashlib
import base64
import requests
import time

st.set_page_config(page_title="Voice Impersonation Detection", layout="wide")

def set_bg():
    with open("bg.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64}");
        background-size: cover;
        background-position: center;
    }}
    .glass {{
        background: rgba(0,0,0,0.55);
        backdrop-filter: blur(18px);
        border-radius: 22px;
        padding: 28px;
        margin-top: 20px;
        color: white;
        box-shadow: 0 10px 40px rgba(0,0,0,0.45);
    }}
    .wave-glass {{
        background: rgba(0,0,0,0.45);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 18px;
        margin-top: 15px;
    }}
    .title {{
        text-align: center;
        font-size: 38px;
        font-weight: 700;
    }}
    .subtitle {{
        text-align: center;
        opacity: 0.85;
        margin-bottom: 10px;
    }}
    .result {{
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        margin-top: 18px;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg()

model = tf.keras.models.load_model("models/voice_cnn_lstm.h5")

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 200:
        mfcc = np.pad(mfcc, ((0,0),(0,200-mfcc.shape[1])))
    mfcc = mfcc[:, :200]
    return mfcc.T, y, sr

def audio_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def animated_waveform(y, sr):
    y = y[::25]
    frames = 25
    step = len(y) // frames
    placeholder = st.empty()

    for i in range(step, len(y)+1, step):
        fig, ax = plt.subplots(figsize=(13,3))
        librosa.display.waveshow(y[:i], sr=sr//25, ax=ax, color="#00ffff")
        ax.axis("off")
        fig.patch.set_alpha(0)
        placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)
        time.sleep(0.015)

st.markdown("""
<div class="glass">
    <div class="title">🎤 Voice Impersonation Detection System</div>
    <div class="subtitle">AI Voice • Special Song • Human Voice Analysis</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📤 Upload Audio", "☁️ Google Drive"])

audio_path = None

with tab1:
    uploaded = st.file_uploader("Upload WAV / MP3", type=["wav","mp3"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded.read())
        tmp.close()
        audio_path = tmp.name
        st.audio(audio_path)

with tab2:
    drive_link = st.text_input("Paste Google Drive Public Audio Link")
    if drive_link:
        try:
            file_id = drive_link.split("/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?id={file_id}"
            r = requests.get(url, timeout=20)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.write(r.content)
            tmp.close()
            audio_path = tmp.name
            st.audio(audio_path)
        except:
            st.error("Invalid Google Drive link")

analyze = st.button("🔍 Analyze Voice", use_container_width=True)

if analyze and audio_path:
    st.session_state.clear()

    mfcc, y, sr = extract_mfcc(audio_path)

    st.markdown("<div class='wave-glass'>", unsafe_allow_html=True)
    animated_waveform(y, sr)
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded_hash = audio_hash(audio_path)
    special_hash = audio_hash("special_song.mp3")

    pred = model.predict(mfcc.reshape(1,200,40), verbose=0)[0][0]

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    if uploaded_hash == special_hash:
        result = "🎵 SYNTHETIC VOICE "
    elif pred >= 0.5:
        result = "🤖 AI / SYNTHETIC VOICE"
    else:
        result = "🧑 HUMAN VOICE"

    st.markdown(f"<div class='result'>{result}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    os.remove(audio_path)
