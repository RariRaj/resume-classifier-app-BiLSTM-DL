ResumeAtlas: AI-Powered Career Classification
Transforming Raw Resumes into Career Insights
ResumeAtlas is a deep learning solution designed to categorize resumes into 40+ professional domains. By utilizing a BiLSTM architecture with a Bahdanau-style Attention Mechanism, the model doesn't just read text—it learns which specific skills and keywords carry the most weight for each unique industry.

🧠 Model Architecture & Innovation
Unlike standard classifiers, this project uses a multi-stage approach to handle industry-specific jargon and "confused" categories (e.g., distinguishing between a Database Administrator and a SQL Developer).

Technical Highlights:
Custom Attention Layer: Implemented a trainable attention mechanism that assigns higher weights to "discriminative" words while ignoring generic resume fluff.

Dual Pooling Strategy: Concatenates Attention-weighted output with Global Max Pooling to capture both contextual importance and the strongest individual feature signals.

Bi-Directional LSTM: Processes text in both forward and backward directions to capture long-term dependencies in professional summaries.

Domain Keyword Boosting: A custom preprocessing pipeline that identifies and amplifies "anchor words" for highly similar job categories.

🛠️ Project Structure
Plaintext
├── notebooks/
│   └── Resume_Category_model.ipynb    # Full training pipeline & EDA
├── assets/
│   ├── resume_classifier.keras        # Trained weights (Deep Learning Model)
│   ├── tokenizer.pkl                  # Fitted word-to-index mapping
│   └── label_encoder.pkl              # ID-to-Category mapping
├── app.py                             # Streamlit Web Application
├── requirements.txt                   # Environment dependencies
└── README.md                          # Project documentation

🏯The Complete Architecture 
What the project does
It takes raw resume text as input and predicts which of 43 job categories (Accountant, SQL Developer, React Developer, etc.) that resume belongs to. This is a multiclass text classification problem solved with deep learning.

Layer 1 — Data Layer
----------------------
The dataset is 13,389 real resumes from HuggingFace with two columns — Text (resume content) and Category (job label). The data has a class imbalance where some categories have 800+ resumes and others have under 100, which the architecture specifically addresses with class weights.

Layer 2 — Preprocessing Architecture
---------------------------------------
This is the most carefully engineered part of the whole project. It has three sub-components working together:
The clean_resume() function first strips all noise using regex — URLs, emails, HTML tags, symbols — leaving only lowercase letters. Then it removes two sets of words: standard English stopwords (the, is, at) and custom resume-generic words (experience, skills, management) that appear identically in every category and give the model zero signal. Finally, if the category is known (during training), it appends domain-specific keywords from DOMAIN_BOOST for categories that were getting confused — for example, appending "tsql plsql stored procedures triggers" to SQL Developer resumes so the model sees more of what makes that category unique.
After cleaning, the Tokenizer builds a vocabulary of 25,000 words and converts every word to an integer. Then pad_sequences makes every resume exactly 300 integers long — short ones get zeros appended, long ones get cut.

Layer 3 — Model Architecture
-------------------------------
The neural network has a branching design with seven distinct stages:
The Embedding layer (25000 → 256 dimensions) converts each of the 300 word integers into a 256-float vector. These vectors are learned — similar words end up near each other in 256-dimensional space. L2 regularisation prevents the embeddings from growing too large.
SpatialDropout1D(0.2) randomly drops 20% of the 256 dimensions across the entire sequence — forcing the model to not rely on any single dimension.
The Bidirectional LSTM(128) reads the 300-word sequence both forward and backward simultaneously. The forward pass reads word 1 to 300, the backward pass reads word 300 to 1. Both outputs are concatenated giving 256 values per timestep. This is crucial because "not guilty" has different meaning depending on direction.
Here the architecture splits into two parallel paths:
The custom AttentionLayer learns which of the 300 word positions matter most for classification. It computes a score for each position using a learned weight matrix W and bias b, applies softmax to get probabilities, then creates a weighted sum. For a SQL Developer resume the attention focuses on "tsql", "stored procedures", "query" and ignores generic words.
GlobalMaxPooling1D takes the maximum value at each of the 256 features across all 300 positions — capturing the strongest signal at any point in the resume regardless of position.
Both outputs are Concatenated to give a 512-dimensional vector, then BatchNormalization stabilises the values. Three Dense layers then act as a funnel: 512 → 256 → 128, each with relu activation, L2 regularisation, Dropout, and BatchNormalization.
The final Dense(43, softmax) layer outputs 43 probabilities summing to 1.0. argmax picks the highest — that is the predicted category.

Layer 4 — Training Architecture
--------------------------------------
Every epoch processes all 10,711 training resumes in batches of 32 (335 gradient updates per epoch). For each batch: forward pass produces predictions, loss is calculated using SparseCategoricalCrossentropy and multiplied by the class weight for rare categories, backpropagation computes gradients, and Adam updates all weights. The three callbacks — EarlyStopping(patience=8), ReduceLROnPlateau(factor=0.4), and ModelCheckpoint — together ensure training stops at the right time, adapts the learning rate intelligently, and always saves the best version.

Layer 5 — Inference Architecture (Two-Pass)
-------------------------------------------------
The predict_resume() function uses a clever two-pass approach. Pass 1 cleans the resume without any boost and gets an initial predicted category. Pass 2 uses that predicted category to look up the domain boost words, appends them to the cleaned text, and runs a second prediction. This gives the model extra discriminating vocabulary at inference time, improving confidence on the categories that were previously confused.

🚀 Deployment (Streamlit)
The model is deployed as a user-friendly web app.

Input: Users can paste raw text from any resume.

Processing: The app applies the same v3 cleaning logic used during training.

Inference: The model returns the top category and a confidence percentage.

How to Run Locally:
Clone the repository:

Bash
git clone https://github.com/RariRaj/ResumeAtlas.git
Install dependencies:

Bash
pip install -r requirements.txt
Launch the app:

Bash
streamlit run app.py
📊 Performance & Optimization
Optimizer: Adam with clipnorm=1.0 to prevent gradient explosion in the attention layer.

Regularization: Utilizes SpatialDropout1D, BatchNormalization, and Label Smoothing to ensure the model generalizes well to unseen resume formats.

Callbacks: Integrated ReduceLROnPlateau and EarlyStopping to optimize the learning curve and prevent overfitting.
