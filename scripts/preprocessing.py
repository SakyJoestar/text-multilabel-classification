import re
import string
import pandas as pd
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  

print("Cargando datos...")
df = pd.read_csv("data/Tweets.csv")

df = df[["airline_sentiment", "text"]]
df = df.rename(columns={"airline_sentiment": "sentiment", "text": "text"})
df.dropna(inplace=True)

sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
df["sentiment_label"] = df["sentiment"].map(sentiment_map)

# --- Configuración para normalización ---
stop_words = set(stopwords.words("english"))
important_words = {"not", "no", "nor", "never"}
stop_words = stop_words - important_words  # NO borrar negaciones

stemmer = PorterStemmer()

emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F700-\U0001F77F"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = emoji_pattern.sub(" EMOJI ", text)
    text = re.sub(r"[^a-z0-9\s!?.,;:()'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return text.split()


def normalize_tokens(tokens):
    norm_tokens = []
    for tok in tokens:
        if tok in stop_words:
            continue
        norm_tokens.append(tok)
    return norm_tokens


def preprocess_text(text: str):
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = normalize_tokens(tokens)
    return tokens


print("Limpiando y preprocesando texto...")
df["cleaned_text"] = df["text"].apply(clean_text)
df["tokens"] = df["cleaned_text"].apply(lambda t: normalize_tokens(tokenize(t)))

