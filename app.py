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
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias", shape=(1,), initializer="zeros", trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()


# --- 2. Load Model and Assets ---
@st.cache_resource
def load_model_and_assets():
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    weights_path = os.path.join(
        curr_dir, "streamlit_assets", "model_weights.weights.h5"
    )
    tokenizer_path = os.path.join(curr_dir, "streamlit_assets", "tokenizer.pkl")
    le_path = os.path.join(curr_dir, "streamlit_assets", "label_encoder.pkl")

    # Reconstruct based on the verified H5 shapes
    model = tf.keras.Sequential(
        [
            layers.Embedding(input_dim=25000, output_dim=256, input_length=300),
            layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
            AttentionLayer(),
            layers.BatchNormalization(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(128, activation="relu"),
            layers.Dense(43, activation="softmax"),
        ]
    )

    model.build(input_shape=(None, 300))

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        st.error("Weights file not found!")
        return None, None, None

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    return model, tokenizer, le


model, tokenizer, le = load_model_and_assets()


# --- 3. Prediction Logic ---
def predict_resume(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join(text.split())

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=300, padding="post", truncating="post")

    prediction = model.predict(padded, verbose=0)
    class_idx = np.argmax(prediction)
    category = le.classes_[class_idx]
    confidence = np.max(prediction) * 100

    return category, confidence


# --- 4. Streamlit UI ---
st.set_page_config(page_title="Resume Classifier", page_icon="📄")
st.title("Advanced Resume Classifier")
st.write(f"Predicting across **{len(le.classes_)}** professional categories.")

resume_text = st.text_area("Paste Resume Text Here:", height=300)

if st.button("Predict Category"):
    if not resume_text:
        st.warning("Please provide text.")
    else:
        with st.spinner("Classifying..."):
            category, confidence = predict_resume(resume_text)
            st.success(f"**Category:** {category}")
            st.info(f"**Confidence:** {confidence:.2f}%")
