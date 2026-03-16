🧠 Duplicate Question Analysis

A Natural Language Processing (NLP) project that detects whether two questions are duplicate or semantically similar using a Deep Learning LSTM model.

This system helps platforms like Q&A websites, educational forums, and customer support systems reduce redundant questions and improve information retrieval.

🚀 Features

✅ Detects duplicate questions using deep learning
✅ Understands semantic similarity between sentences
✅ Built using LSTM neural networks
✅ Text preprocessing pipeline included
✅ Trained on 20,000 question pairs
✅ Achieved 92% accuracy


⚙️ Tech Stack
Category	Tools
Programming	Python
Deep Learning	TensorFlow / Keras
NLP	NLTK
Data Processing	Pandas, NumPy
Visualization	Matplotlib
🧩 Workflow
1️⃣ Data Collection

Dataset containing pairs of questions with labels indicating whether they are duplicates.

2️⃣ Data Preprocessing

Text cleaning steps:

Lowercasing

Removing punctuation

Tokenization

Stopword removal

Padding sequences

3️⃣ Feature Representation

Questions are converted into numerical vectors using tokenization and embeddings.

4️⃣ Model Training

A Long Short-Term Memory (LSTM) network is trained to capture semantic relationships between question pairs.

5️⃣ Prediction

The trained model predicts whether two questions are:

Duplicate

Not Duplicate

📊 Model Performance
Metric	Value
Accuracy	92%
Dataset Size	20,000 Question Pairs
Model	LSTM
