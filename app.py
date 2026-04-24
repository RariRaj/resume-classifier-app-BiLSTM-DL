import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Build the path
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(curr_dir, "streamlit_assets", "resume_classifier.h5")


import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re


# Define the custom AttentionLayer (must be available when loading the model)
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
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()


@st.cache_resource
def load_model_and_assets():
    # Debug: Check if file exists and its size
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"DEBUG: Model found. Size: {size_mb:.2f} MB")
    else:
        st.error(f"Model file NOT found at: {model_path}")
        return None, None, None

    # Load using the .h5 file
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,  # Adding this ensures it doesn't fail on optimizer states
    )

    # 2. Load the Tokenizer
    with open("streamlit_assets/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # 3. Load the Label Encoder
    with open("streamlit_assets/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return model, tokenizer, le


model, tokenizer, le = load_model_and_assets()


def predict_new_resume(text):
    # Basic Cleaning (Matches your training preprocessing)
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join(text.split())

    # 1. Convert text to sequence of numbers
    sequence = tokenizer.texts_to_sequences([text])

    # 2. Pad to the same length used in training (300)
    padded = pad_sequences(sequence, maxlen=300, padding="post", truncating="post")

    # 3. Predict
    prediction = model.predict(padded, verbose=0)

    # 4. Get the category name
    class_idx = np.argmax(prediction)
    category = le.classes_[class_idx]
    confidence = np.max(prediction) * 100

    return category, confidence


# Streamlit UI
st.title("Resume Category Classifier")
st.write("Upload a resume or paste text to predict its category.")

# Input text area for resume
resume_text = st.text_area("Paste Resume Text Here:", height=300)

if st.button("Predict Category"):
    if resume_text:
        category, confidence = predict_new_resume(resume_text)
        st.success(f"**Predicted Category:** {category}")
        st.info(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("Please paste some resume text to get a prediction.")
