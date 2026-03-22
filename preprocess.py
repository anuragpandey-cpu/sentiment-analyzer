"""
preprocess.py
Text cleaning and preprocessing utilities for Sentiment Analyzer.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep some negation words that affect sentiment
KEEP_WORDS = {'not', 'no', 'never', 'neither', 'nor', "n't"}
stop_words -= KEEP_WORDS


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
    - Lowercase
    - Remove URLs, emails, special characters
    - Tokenize
    - Remove stopwords (keeping negations)
    - Lemmatize
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Expand common contractions
    contractions = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'ve": " have", "'m": " am"
    }
    for key, val in contractions.items():
        text = text.replace(key, val)

    # Remove special characters (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = text.split()

    # Remove short tokens (<= 1 char) and stopwords, then lemmatize
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if len(w) > 1 and w not in stop_words
    ]

    return ' '.join(tokens)


def batch_clean(texts: list) -> list:
    """Clean a list of texts."""
    return [clean_text(t) for t in texts]
