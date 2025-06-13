import yaml
import os
import re
import spacy
import importlib.util
from spacy.cli import download


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def model_loading():
    # Check if the model is already installed
    model_name = "en_core_web_sm"
    if importlib.util.find_spec(model_name) is None:
        download(model_name)

    nlp = spacy.load(model_name)
    return nlp

def preprocess_text(text: str,nlp) -> str:
    try:
        if not isinstance(text, str):
            return ""

    # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

    # Remove Markdown images and HTML images
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Markdown-style ![alt](url)
        text = re.sub(r'<img[^>]*>', '', text)       # HTML-style <img ... >

    # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

    # Remove hashtags
        text = re.sub(r'#\w+', '', text)

    # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

    # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)

    # NLP: lowercase, remove stopwords, lemmatize
        doc = nlp(text.lower())
        cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

        return ' '.join(cleaned_tokens)

    except Exception as e:
        raise e

