import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Build the path
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(curr_dir, "streamlit_assets", "resume_classifier.h5")


import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re


# Resolve potential protobuf conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# --- 1. Define Custom Attention Layer ---
# This must match the class definition used during training
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias", shape=(1,), initializer="zeros", trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Attention Mechanism: e = tanh(Wx + b)
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()

    # --- 2. Load Model and Assets ---@st.cache_resource


def load_model_and_assets():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(
        curr_dir, "streamlit_assets", "model_weights.weights.h5"
    )

    # 1. Reconstruct EXACTLY based on your H5 shapes
    model = tf.keras.Sequential(
        [
            # Shape: (25000, 256)
            layers.Embedding(input_dim=25000, output_dim=256, input_length=300),
            # LSTM Shape: (256, 512) -> 512/4 = 128 units
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            # Attention Layer Shape: (256, 1) -> Expects 256 features from Bi-LSTM
            AttentionLayer(),
            # Batch Normalization (512 features from Bi-LSTM 128*2)
            layers.BatchNormalization(),
            # Dense 1 Shape: (512, 512)
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            # Dense 2 Shape: (512, 256)
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            # Dense 3 Shape: (256, 128)
            layers.Dense(128, activation="relu"),
            # Final Output Shape: (128, 43) -> 43 Categories
            layers.Dense(43, activation="softmax"),
        ]
    )

    # 2. Build the model
    model.build(input_shape=(None, 300))

    # 3. Load weights
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        st.error("Weights file not found!")
        return None, None, None

    # Load the Tokenizer
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        st.error(f"Tokenizer not found at: {tokenizer_path}")
        return None, None, None

    # Load the Label Encoder
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le = pickle.load(f)
    else:
        st.error(f"Label Encoder not found at: {le_path}")
        return None, None, None

    return model, tokenizer, le


# Initialize model, tokenizer, and label encoder
model, tokenizer, le = load_model_and_assets()


# --- 3. Prediction Logic ---
def predict_new_resume(text):
    # Basic Cleaning (Matches standard text preprocessing)
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join(text.split())

    # Tokenize and Pad to maxlen=300
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=300, padding="post", truncating="post")

    # Run Inference
    prediction = model.predict(padded, verbose=0)

    # Extract predicted class and confidence
    class_idx = np.argmax(prediction)
    category = le.classes_[class_idx]
    confidence = np.max(prediction) * 100

    return category, confidence


# --- 4. Streamlit UI ---
st.set_page_config(page_title="Resume Classifier", page_icon="📄")
st.title("Resume Category Classifier")
st.write("Determine the professional category of a resume using BiLSTM + Attention.")

# Input text area
resume_text = st.text_area("Paste Resume Text Here:", height=300)

if st.button("Predict Category"):
    if not resume_text:
        st.warning("Please provide some text to analyze.")
    elif model is None:
        st.error(
            "Assets failed to load. Please check the logs and your 'streamlit_assets' folder."
        )
    else:
        with st.spinner("Analyzing text..."):
            try:
                category, confidence = predict_new_resume(resume_text)
                st.success(f"**Predicted Category:** {category}")
                st.info(f"**Confidence:** {confidence:.2f}%")
            except Exception as e:
                st.error(f"Prediction error: {e}")
