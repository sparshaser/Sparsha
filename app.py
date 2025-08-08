import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import librosa
import librosa.display
import joblib
from scipy.io.wavfile import write
from sklearn.metrics import confusion_matrix, accuracy_score
from model import CNN_GRU_Attention
from utils import extract_features

# Constants
SAVE_DIR = "outputs"
MY_DATASET_DIR = "hindi_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("same_output", exist_ok=True)
os.makedirs("misclassified_output", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, scaler, encoder
@st.cache_resource
def load_model_and_utils():
    encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    model = CNN_GRU_Attention(input_dim=374, num_classes=len(encoder.classes_))
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, scaler, encoder

model, scaler, encoder = load_model_and_utils()

# Predict emotion
def predict_emotion(audio, sr):
    x = extract_features(audio, sr).reshape(1, -1)
    x = scaler.transform(x)
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        output, _ = model(x)
        prob = torch.softmax(output[0], dim=0).cpu().numpy()
        pred = encoder.inverse_transform([np.argmax(prob)])[0]
    return pred, prob

# sparsha importance
def sparsha_importance(y, baseline_prob, sr):
    stride = int(0.1 * sr)
    win = int(0.5 * sr)
    pred_idx = np.argmax(baseline_prob)
    importance = []
    for start in range(0, len(y), stride):
        end = min(start + win, len(y))
        y_masked = y.copy()
        y_masked[start:end] = 0.0
        _, prob_masked = predict_emotion(y_masked, sr)
        diff = baseline_prob[pred_idx] - prob_masked[pred_idx]
        importance.append(diff)
    importance = np.array(importance)
    total = np.sum(importance)
    return (importance / total) * 100 if total > 0 else importance

# Save visuals and masked audio
def plot_and_save_all(y, sr, pred, importance, true_label, filename):
    is_correct = (pred.lower() == true_label.lower())
    output_dir = os.path.join("same_output" if is_correct else "misclassified_output")
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{true_label}_{pred}_{filename}"

    # Waveform
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    plt.tight_layout()
    path1 = os.path.join(output_dir, f"{fname}_waveform.png")
    plt.savefig(path1); st.image(path1); plt.close()

    # Spectrogram
    fig2, ax2 = plt.subplots()
    S = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz", ax=ax2)
    ax2.set_title("Spectrogram")
    plt.tight_layout()
    path2 = os.path.join(output_dir, f"{fname}_spectrogram.png")
    plt.savefig(path2); st.image(path2); plt.close()

    # sparsha Bar Plot
    fig3, ax3 = plt.subplots()
    t = np.linspace(0, len(importance) * 0.1, len(importance))
    ax3.bar(t, importance, width=0.09, color='red')
    ttl = "Actual: " + true_label + " Predicted: " + pred + " Importance (%)"
    ax3.set_title(ttl)
    ax3.set_xlabel("Time (s)")
    plt.tight_layout()
    path3 = os.path.join(output_dir, f"{fname}_explanation.png.png")
    plt.savefig(path3); st.image(path3); plt.close()

    # Masked audio: mute low-importance segments
    stride = int(0.1 * sr)
    win = int(0.5 * sr)
    max_score = np.max(importance)
    threshold = 0.6 * max_score
    masked_audio = np.zeros_like(y)

    for i, score in enumerate(importance):
        if score >= threshold:
            start = i * stride
            end = min(start + win, len(masked_audio))
            masked_audio[start:end] = y[start:end]

    wav_path = os.path.join(output_dir, f"{fname}_masked.wav")
    write(wav_path, sr, masked_audio.astype(np.float32))

# --- STREAMLIT UI ---
st.title("🎤 Real-Time Emotion Detection App with sparsha")

option = st.radio("Select Input Method", ["Upload File", "Record Audio", "Run Test Dataset"])

# Upload File
if option == "Upload File":
    uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
    if uploaded_file:
        path = "uploaded.wav"
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        y, sr = librosa.load(path, sr=16000)
        pred, prob = predict_emotion(y, sr)
        importance = sparsha_importance(y, prob, sr)
        st.success(f"🎯 Predicted Emotion: `{pred}`")
        plot_and_save_all(y, sr, pred, importance, "unknown", "upload")

# Record Audio
elif option == "Record Audio":
    duration = st.selectbox("🎚️ Select Duration (seconds)", options=[5, 10, 20], index=1)
    if st.button("Start Recording"):
        import sounddevice as sd
        st.info("🎙️ Recording...")
        audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        y = audio.flatten()
        sr = 16000
        pred, prob = predict_emotion(y, sr)
        importance = sparsha_importance(y, prob, sr)
        st.success(f"🎯 Predicted Emotion: `{pred}`")
        plot_and_save_all(y, sr, pred, importance, "unknown", "recorded")

# Run Test Dataset
elif option == "Run Test Dataset":
    actual, predicted = [], []
    for root, _, files in os.walk(MY_DATASET_DIR):
        for f in files:
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                y, sr = librosa.load(path, sr=16000)
                pred, prob = predict_emotion(y, sr)
                true = os.path.basename(os.path.dirname(path)).lower()
                actual.append(true)
                predicted.append(pred)

                importance = sparsha_importance(y, prob, sr)
                plot_and_save_all(y, sr, pred, importance, true, f.replace(".wav", ""))

                if pred != true:
                    st.markdown(f"⚠️ Mismatch: `{f}` | True: `{true}` | Pred: `{pred}`")
                else:
                    st.markdown(f"✅ Correct: `{f}` | Emotion: `{true}`")

    # Confusion Matrix
    cm = confusion_matrix(actual, predicted, labels=encoder.classes_)
    acc = accuracy_score(actual, predicted)
    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    cm_path = f"{SAVE_DIR}/confusion_matrix.png"
    plt.savefig(cm_path)
    st.image(cm_path)
    st.success(f"Overall Accuracy: **{acc * 100:.2f}%**")

