# app/model/preprocessing.py
import re
from nltk.corpus import stopwords
import nltk # Add this import

# Ensure you have the NLTK stopwords downloaded
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by removing non-alphabetic characters,
    converting to lowercase, and removing stopwords.
    Args:
        text (str): The input text to preprocess.
    Returns:
        str: The preprocessed text.
    """

    # 1. Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 2. Lowercase
    text = text.lower()
    # 3. Tokenize & filter stopwords
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)
