"""
train.py
Training script for the Student Feedback Sentiment Analyzer.

Dataset CSV format (data/feedback.csv):
    feedback  : Raw text feedback from students
    sentiment : Label string — 'Positive', 'Neutral', or 'Negative'

Usage:
    python train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocess import batch_clean
from model import build_model

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CONFIG = {
    'data_path'    : 'data/feedback.csv',
    'model_dir'    : 'model',
    'max_words'    : 10000,      # Vocabulary size
    'max_len'      : 100,        # Max sequence length
    'embedding_dim': 128,
    'epochs'       : 15,
    'batch_size'   : 32,
    'test_size'    : 0.2,
    'val_size'     : 0.1,
    'random_state' : 42,
}

LABELS = ['Negative', 'Neutral', 'Positive']
LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}


def load_and_prepare_data(path: str):
    """Load CSV, clean text, encode labels."""
    print("📂 Loading dataset...")
    df = pd.read_csv(path)

    # Validate columns
    assert 'feedback' in df.columns and 'sentiment' in df.columns, \
        "CSV must have 'feedback' and 'sentiment' columns"

    df.dropna(subset=['feedback', 'sentiment'], inplace=True)
    df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
    df = df[df['sentiment'].isin(LABELS)].reset_index(drop=True)

    print(f"✅ Dataset loaded: {len(df)} samples")
    print(df['sentiment'].value_counts().to_string())

    print("\n🔧 Cleaning text...")
    df['cleaned'] = batch_clean(df['feedback'].tolist())
    df['label'] = df['sentiment'].map(LABEL_MAP)

    return df


def tokenize(texts, max_words, oov_token='<OOV>'):
    """Fit tokenizer and convert texts to sequences."""
    tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return tokenizer, sequences


def plot_history(history, save_dir):
    """Save training accuracy and loss plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print("📊 Training plots saved.")


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("📊 Confusion matrix saved.")


def main():
    os.makedirs(CONFIG['model_dir'], exist_ok=True)

    # Load data
    df = load_and_prepare_data(CONFIG['data_path'])

    # Tokenize
    print("\n🔤 Tokenizing...")
    tokenizer, sequences = tokenize(df['cleaned'].tolist(), CONFIG['max_words'])
    vocab_size = min(CONFIG['max_words'], len(tokenizer.word_index) + 1)

    # Pad sequences
    X = pad_sequences(sequences, maxlen=CONFIG['max_len'],
                      padding='post', truncating='post')
    y = to_categorical(df['label'].values, num_classes=3)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=df['label'].values
    )

    # Further split train into train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )

    print(f"\n📐 Shapes — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Build model
    print("\n🏗️  Building model...")
    model = build_model(vocab_size, CONFIG['embedding_dim'], CONFIG['max_len'])
    model.summary()

    # Callbacks
    model_path = os.path.join(CONFIG['model_dir'], 'sentiment_model.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    # Train
    print("\n🚀 Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    # Save tokenizer
    tok_path = os.path.join(CONFIG['model_dir'], 'tokenizer.pkl')
    with open(tok_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"\n💾 Tokenizer saved → {tok_path}")

    # Evaluate
    print("\n📈 Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss     : {loss:.4f}")
    print(f"   Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")

    # Classification report
    y_pred = model.predict(X_test)
    y_pred_labels = y_pred.argmax(axis=1)
    y_true_labels = y_test.argmax(axis=1)
    print("\n📋 Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=LABELS))

    # Plots
    plot_history(history, CONFIG['model_dir'])
    plot_confusion_matrix(y_true_labels, y_pred_labels, CONFIG['model_dir'])

    print(f"\n✅ Training complete! Model saved → {model_path}")


if __name__ == '__main__':
    main()
