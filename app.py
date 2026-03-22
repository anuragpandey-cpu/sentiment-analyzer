"""
app.py
Flask web application for Student Feedback Sentiment Analyzer.

Run:
    python app.py

Endpoints:
    GET  /           → Render main UI
    POST /predict    → JSON sentiment prediction
    GET  /health     → Health check
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# ─── Suppress TF verbose logs ───────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH     = os.path.join('model', 'sentiment_model.h5')
TOKENIZER_PATH = os.path.join('model', 'tokenizer.pkl')
MAX_LEN        = 100
LABELS         = ['Negative', 'Neutral', 'Positive']

# Emoji and color mapping for each sentiment
SENTIMENT_META = {
    'Positive': {'emoji': '😊', 'color': '#22c55e', 'icon': 'thumb-up'},
    'Neutral' : {'emoji': '😐', 'color': '#f59e0b', 'icon': 'minus'},
    'Negative': {'emoji': '😔', 'color': '#ef4444', 'icon': 'thumb-down'},
}

# ─────────────────────────────────────────
# LOAD MODEL & TOKENIZER
# ─────────────────────────────────────────
print("🔄 Loading model and tokenizer...")
try:
    model     = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️  Model not found: {e}")
    print("   Run 'python train.py' first to train the model.")
    model     = None
    tokenizer = None


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def index():
    """Render main UI page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render About Us page."""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Render Contact Us page."""
    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "text": "Student feedback text here" }
    Returns: JSON with sentiment, confidence, probabilities
    """
    if model is None or tokenizer is None:
        return jsonify({
            'error': 'Model not loaded. Please run train.py first.'
        }), 503

    data = request.get_json(force=True)
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Please provide non-empty feedback text.'}), 400

    if len(text) > 2000:
        return jsonify({'error': 'Text too long. Max 2000 characters.'}), 400

    # Preprocess
    cleaned  = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    # Predict
    prediction   = model.predict(padded, verbose=0)[0]
    pred_index   = int(np.argmax(prediction))
    sentiment    = LABELS[pred_index]
    confidence   = float(np.max(prediction))

    probabilities = {
        label: round(float(prediction[i]) * 100, 2)
        for i, label in enumerate(LABELS)
    }

    return jsonify({
        'sentiment'    : sentiment,
        'confidence'   : round(confidence * 100, 2),
        'probabilities': probabilities,
        'emoji'        : SENTIMENT_META[sentiment]['emoji'],
        'color'        : SENTIMENT_META[sentiment]['color'],
        'cleaned_text' : cleaned,
        'word_count'   : len(text.split())
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status'      : 'ok',
        'model_loaded': model is not None
    })


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
