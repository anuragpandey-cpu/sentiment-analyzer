# 🎓 Student Feedback Sentiment Analyzer

A deep learning-based web application that classifies student feedback into **Positive**, **Neutral**, or **Negative** sentiment using a Bidirectional LSTM model built with TensorFlow/Keras and deployed via Flask.

---

## 📁 Project Structure

```
sentiment_analyzer/
├── app.py              → Flask web application
├── model.py            → Bidirectional LSTM architecture
├── train.py            → Model training pipeline
├── preprocess.py       → Text cleaning & preprocessing utilities
├── requirements.txt    → Python dependencies
├── data/
│   └── feedback.csv    → Sample labeled dataset
├── model/              → Saved model files (generated after training)
│   ├── sentiment_model.h5
│   └── tokenizer.pkl
└── templates/
    └── index.html      → Frontend UI (dark theme, animated)
```

---

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
The `data/feedback.csv` file should have two columns:
```
feedback, sentiment
"The teacher is very clear.", Positive
"Classes are okay.", Neutral
"Very poor explanation.", Negative
```
Minimum 500–1000 samples recommended for good accuracy.

### 3. Train the model
```bash
python train.py
```
This saves:
- `model/sentiment_model.h5`  → Trained LSTM weights
- `model/tokenizer.pkl`       → Fitted tokenizer
- `model/training_history.png`
- `model/confusion_matrix.png`

### 4. Run the web app
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## 🚀 Features

| Feature                  | Description                                      |
|--------------------------|--------------------------------------------------|
| Real-time prediction     | Instant results after clicking Analyze           |
| Probability bars         | Visual confidence scores for all 3 classes       |
| Sample inputs            | Pre-loaded example sentences for quick testing   |
| Analysis history         | Tracks last 5 predictions in the session         |
| Keyboard shortcut        | `Ctrl+Enter` to trigger analysis                 |
| Responsive UI            | Works on desktop and mobile                      |

---

## 🧠 Model Architecture

```
Input Text
    ↓
Preprocessing (NLTK: clean, tokenize, lemmatize)
    ↓
Embedding Layer (128-dim)
    ↓
SpatialDropout1D (0.2)
    ↓
Bidirectional LSTM (128 units)
    ↓
Bidirectional LSTM (64 units)
    ↓
BatchNormalization
    ↓
Dense(64, ReLU) → Dropout(0.3)
    ↓
Dense(3, Softmax) → [Negative, Neutral, Positive]
```

---

## 📊 API Endpoints

| Method | Endpoint   | Description         |
|--------|------------|---------------------|
| GET    | `/`        | Main UI page        |
| POST   | `/predict` | Sentiment analysis  |
| GET    | `/health`  | Server health check |

### Example API call
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The faculty is very helpful and explains well."}'
```

### Response
```json
{
  "sentiment": "Positive",
  "confidence": 94.32,
  "probabilities": {
    "Negative": 1.23,
    "Neutral": 4.45,
    "Positive": 94.32
  },
  "emoji": "😊",
  "color": "#22c55e",
  "word_count": 9
}
```

---

## 🛠️ Technologies Used

| Category        | Technology            |
|-----------------|-----------------------|
| Deep Learning   | TensorFlow, Keras     |
| NLP             | NLTK                  |
| Backend         | Flask (Python)        |
| Frontend        | HTML, CSS, JavaScript |
| Data Processing | Pandas, NumPy         |
| Visualization   | Matplotlib, Seaborn   |

---

## 👨‍💻 Minor Project | B.Tech CSE
