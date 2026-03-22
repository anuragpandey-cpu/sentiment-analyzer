"""
model.py
Bidirectional LSTM model architecture for Sentiment Classification.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    Bidirectional, SpatialDropout1D, BatchNormalization
)
from tensorflow.keras.regularizers import l2


def build_model(vocab_size: int, embedding_dim: int = 128,
                max_len: int = 100, num_classes: int = 3) -> Sequential:
    """
    Build and compile Bidirectional LSTM model.

    Architecture:
        Embedding → SpatialDropout → BiLSTM (128) → BiLSTM (64)
        → BatchNorm → Dense(64, relu) → Dropout → Dense(3, softmax)

    Args:
        vocab_size     : Total vocabulary size (including OOV token)
        embedding_dim  : Dimensionality of word embeddings
        max_len        : Maximum sequence length after padding
        num_classes    : Number of sentiment classes (default: 3)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        # Word Embedding Layer
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            name='embedding'
        ),

        # Spatial Dropout to reduce overfitting on embedding
        SpatialDropout1D(0.2, name='spatial_dropout'),

        # First BiLSTM layer – captures context from both directions
        Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name='bilstm_1'
        ),

        # Second BiLSTM layer – deeper feature extraction
        Bidirectional(
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            name='bilstm_2'
        ),

        # Batch Normalization for stable training
        BatchNormalization(name='batch_norm'),

        # Fully connected layer
        Dense(64, activation='relu', kernel_regularizer=l2(0.001), name='dense_1'),
        Dropout(0.3, name='dropout'),

        # Output layer (Softmax for multi-class)
        Dense(num_classes, activation='softmax', name='output')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def model_summary_str(model) -> str:
    """Return model summary as a string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return '\n'.join(lines)
