# Sparsha
This repository is created to share the implementation of our posthoc explanatory framework SPARSHA for low resource Speech Emotion Recognition.
SPARSHA – Speech Emotion Recognition with Posthoc Listenable Explanations
This repository contains the implementation of SPARSHA, a posthoc, model-agnostic explanation framework for Speech Emotion Recognition (SER).
It not only predicts emotions from speech but also highlights and plays back the most salient audio segments influencing the prediction — making explanations listenable and accessible.

📌 Features
Real-time Emotion Prediction from .wav files or microphone recordings

Model-Agnostic Posthoc Explanation via perturbation-based masking

Listenable Explanations: hear the parts of the speech driving the decision

Data Augmentation: pitch shift, time stretch, noise addition

Custom Deep Model: CNN + GRU + Attention for temporal-spectral feature modeling

Evaluation Tools: confusion matrix, accuracy scores, and explanation visualizations

🗂 Project Structure
bash
Copy
Edit
.
├── app.py               # Streamlit application for real-time SER and explanations
├── model.py             # CNN-GRU-Attention model architecture
├── utils.py             # Audio preprocessing, augmentation, and feature extraction
├── requirements.txt     # Dependencies list
├── label_encoder.pkl    # Pre-trained label encoder
├── scaler.pkl           # Pre-trained feature scaler
├── best_model.pt        # Trained model weights
└── dataset/       # Dataset folder (organised by emotion classes)
⚙️ Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/sparsha-ser.git
cd sparsha-ser
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Ensure model & utils files are present
Place label_encoder.pkl, scaler.pkl, and best_model.pt in the project root.

🚀 Usage
Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
Modes Available in UI:
Upload File – Upload a .wav file for prediction and explanation

Record Audio – Record speech live via microphone

Run Test Dataset – Process all .wav files in hindi_dataset/ and generate explanations

📊 Outputs Generated
Waveform of the speech

Spectrogram

Explanation Bar Plot showing segment importance

Masked Audio containing only salient emotional cues

Confusion Matrix for batch runs

Correctly classified samples are saved in same_output/
Misclassified samples are saved in misclassified_output/

🧠 Model Details
Input Features: 374-dimensional vector from MFCCs, deltas, mel spectrogram, chroma, contrast, tonnetz, pitch, and energy features

Architecture:

CNN for initial feature extraction

GRU for temporal modeling

Attention mechanism for focusing on salient time steps

Classes Supported: 8 emotions (Anger, Disgust, Fear, Happy, Neutral, Sad, Sarcastic, Surprise)


Results can be accessed here

