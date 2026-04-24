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


# --- 2. Load Model and Assets ---
@st.cache_resource
def load_model_and_assets():
    # Use absolute paths to ensure Streamlit Cloud finds the files
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    weights_path = os.path.join(
        curr_dir, "streamlit_assets", "model_weights.weights.h5"
    )
    tokenizer_path = os.path.join(curr_dir, "streamlit_assets", "tokenizer.pkl")
    le_path = os.path.join(curr_dir, "streamlit_assets", "label_encoder.pkl")

    # Reconstruct the exact architecture used in training
    # input_dim: 25000, output_dim: 256, input_length: 300
    model = tf.keras.Sequential(
        [
            layers.Embedding(input_dim=25000, output_dim=256, input_length=300),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            AttentionLayer(),
            layers.Dense(32, activation="relu"),
            layers.Dense(
                25, activation="softmax"
            ),  # Ensure 25 matches your training classes
        ]
    )

    # Load weights into the reconstructed model
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        st.error(f"Weights file NOT found at: {weights_path}")
        return None, None, None

    # Load the Tokenizer
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        st.error("Tokenizer not found!")
        return None, None, None

    # Load the Label Encoder
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le = pickle.load(f)
    else:
        st.error("Label Encoder not found!")
        return None, None, None

    return model, tokenizer, le


# Initialize everything
model, tokenizer, le = load_model_and_assets()


# --- 3. Prediction Logic ---
def predict_new_resume(text):
    # Preprocessing
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join(text.split())

    # Tokenize and Pad
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=300, padding="post", truncating="post")

    # Run Inference
    prediction = model.predict(padded, verbose=0)

    # Extract results
    class_idx = np.argmax(prediction)
    category = le.classes_[class_idx]
    confidence = np.max(prediction) * 100

    return category, confidence


# --- 4. Streamlit UI ---
st.set_page_config(page_title="Resume Classifier", page_icon="📄")
st.title("Resume Category Classifier")
st.write("Upload or paste a resume to determine its professional category.")

resume_text = st.text_area("Paste Resume Text Here:", height=300)

if st.button("Predict Category"):
    if not resume_text:
        st.warning("Please provide some text.")
    elif model is None:
        st.error(
            "Model initialization failed. Please check file paths in 'streamlit_assets'."
        )
    else:
        with st.spinner("Analyzing resume structure..."):
            category, confidence = predict_new_resume(resume_text)
            st.success(f"**Predicted Category:** {category}")
            st.info(f"**Confidence:** {confidence:.2f}%")
