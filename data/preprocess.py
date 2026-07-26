"""
Text Preprocessing Pipeline
Covers: lowercasing, punctuation removal, stopword removal, lemmatization
Applied to both sentiment analysis and chatbot training data
"""

import re
import string
import nltk

# Download required NLTK data
def download_nltk_data():
    for resource in ['stopwords', 'wordnet', 'punkt', 'omw-1.4']:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove punctuation and special characters
    4. Tokenize (split by whitespace)
    5. Remove stopwords
    6. Lemmatize
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3. Remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    # 4. Tokenize
    tokens = text.split()

    # 5. Remove stopwords
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 1]

    # 6. Lemmatize
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


def batch_preprocess(texts: list) -> list:
    """Preprocess a list of texts."""
    return [preprocess_text(t) for t in texts]
