#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_transformer_sentiment.py

Fine-tuning de un modelo Transformer (DistilBERT) para clasificación
de sentimiento sobre el dataset "Twitter US Airline Sentiment Dataset".

Asume un CSV en data/Tweets.csv con las columnas:
- airline_sentiment  (negative / neutral / positive)
- text               (tweet)

Guarda resultados y modelo en results/TRANSFORMER/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------- CONFIG ----------------------

MODEL_NAME = "distilbert-base-uncased"  # puedes probar otros después
MAX_LEN = 64                            # tweets son cortos, 64 basta
RESULTS_DIR = os.path.join("results", "TRANSFORMER")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# ---------------------- CARGA DE DATOS ----------------------

print("Cargando datos...")
df = pd.read_csv("data/Tweets.csv")

# Nos quedamos solo con lo que interesa
df = df[["airline_sentiment", "text"]].dropna()

# Mapeo de etiquetas a valores numéricos
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["airline_sentiment"].map(label_map)

print(df.head())

# ---------------------- TRAIN / VAL / TEST SPLIT ----------------------

# Primero separamos train+temp y test
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

# Luego de temp sacamos val y test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"],
)

print(f"Tamaño train: {len(train_df)}")
print(f"Tamaño val:   {len(val_df)}")
print(f"Tamaño test:  {len(test_df)}")

# ---------------------- TOKENIZER ----------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------------- DATASET ----------------------


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


train_dataset = TweetDataset(
    train_df["text"], train_df["label"], tokenizer, max_len=MAX_LEN
)
val_dataset = TweetDataset(
    val_df["text"], val_df["label"], tokenizer, max_len=MAX_LEN
)
test_dataset = TweetDataset(
    test_df["text"], test_df["label"], tokenizer, max_len=MAX_LEN
)

# ---------------------- MODELO ----------------------

num_labels = 3  # negative, neutral, positive
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
)

model.to(DEVICE)

# ---------------------- MÉTRICAS ----------------------


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
    }


# ---------------------- TRAINING ARGUMENTS ----------------------

training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=os.path.join(RESULTS_DIR, "logs"),
    logging_steps=50,
    seed=42
)
# ---------------------- TRAINER ----------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ---------------------- ENTRENAMIENTO ----------------------

print("Iniciando entrenamiento...")
trainer.train()

# ---------------------- EVALUACIÓN EN TEST ----------------------

print("\nEvaluando en el conjunto de TEST...")
test_results = trainer.evaluate(test_dataset)
print("Resultados en test:", test_results)

# También sacamos un classification_report bonito
predictions = trainer.predict(test_dataset)
test_logits = predictions.predictions
test_labels = predictions.label_ids
test_preds = np.argmax(test_logits, axis=-1)

report = classification_report(
    test_labels,
    test_preds,
    target_names=["negative", "neutral", "positive"],
)
print("\nClassification Report (TEST):\n")
print(report)

# Guardar el reporte en un txt
report_path = os.path.join(RESULTS_DIR, "classification_report_test.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

# ---------------------- GUARDAR MODELO Y TOKENIZER ----------------------

print("\nGuardando modelo y tokenizer...")
model.save_pretrained(os.path.join(RESULTS_DIR, "best_model"))
tokenizer.save_pretrained(os.path.join(RESULTS_DIR, "best_model"))

print("Listo. Modelo guardado en:", os.path.join(RESULTS_DIR, "best_model"))

# ---------------------- FUNCIÓN DE INFERENCIA ----------------------


def predict_sentiment(text: str, model_path: str = None):
    """
    Carga el modelo guardado (si se especifica model_path) y devuelve
    la etiqueta de sentimiento para un tweet.
    """

    if model_path is None:
        # Usar el modelo actual en memoria
        local_model = model
        local_tokenizer = tokenizer
    else:
        local_tokenizer = AutoTokenizer.from_pretrained(model_path)
        local_model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        ).to(DEVICE)

    encoding = local_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = local_model(**encoding)
        logits = outputs.logits
        pred_label_id = int(torch.argmax(logits, dim=-1).cpu().item())

    inv_label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return inv_label_map[pred_label_id]


if __name__ == "__main__":
    ejemplo = "@united Thanks for the great flight and friendly staff!"
    print("\nEjemplo de predicción:")
    print(ejemplo)
    print("Sentiment:", predict_sentiment(ejemplo))