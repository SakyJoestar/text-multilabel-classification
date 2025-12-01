#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_rnn.py

Entrenamiento de una RNN vanilla para clasificación de sentimiento
sobre el dataset "Twitter US Airline Sentiment Dataset" preprocesado
en preprocessing.py

Guarda resultados en results/RNN/
"""

import os
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt

# Importamos el DataFrame ya preprocesado
from scripts.preprocessing import df  # df tiene: sentiment_label, tokens, etc.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# Carpeta de resultados
RESULTS_DIR = os.path.join("results", "RNN")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------- SEED ----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------- VOCABULARIO ----------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

def build_vocab(token_series, min_freq: int = 1):
    """
    Construye vocabulario a partir de una serie de listas de tokens.
    min_freq: frecuencia mínima para incluir una palabra en el vocab.
    """
    counter = Counter()
    for tokens in token_series:
        counter.update(tokens)

    idx2word = [PAD_TOKEN, UNK_TOKEN]
    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}

    for word, freq in counter.items():
        if freq >= min_freq:
            if word not in word2idx:
                word2idx[word] = len(idx2word)
                idx2word.append(word)

    print(f"Vocabulario construido. Tamaño: {len(idx2word)} palabras.")
    return word2idx, idx2word


def tokens_to_ids(tokens, word2idx):
    return [word2idx.get(tok, UNK_IDX) for tok in tokens]


def pad_sequence(ids, max_len: int, pad_value: int = PAD_IDX):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        return ids + [pad_value] * (max_len - len(ids))


# ---------------------- DATASET ----------------------
class SentimentDataset(Dataset):
    def __init__(self, data_df, word2idx, max_len: int):
        self.word2idx = word2idx
        self.max_len = max_len

        self.labels = data_df["sentiment_label"].values
        self.token_lists = data_df["tokens"].values

        self.sequences = []
        for tokens in self.token_lists:
            ids = tokens_to_ids(tokens, self.word2idx)
            ids = pad_sequence(ids, self.max_len, PAD_IDX)
            self.sequences.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label


# ---------------------- MODELO RNN (VANILLA) ----------------------
class SentimentRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super(SentimentRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX
        )

        # RNN vanilla (sin memoria tipo LSTM/GRU)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            hidden_dim * (2 if bidirectional else 1),
            num_classes
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        embedded = self.embedding(x)          # [B, L, E]
        output, hidden = self.rnn(embedded)   # hidden: [num_layers * num_dir, B, H]

        # Usamos el último hidden state
        if self.bidirectional:
            # Concatenamos la última capa de forward y backward
            # hidden shape: [num_layers*2, B, H]
            fwd_last = hidden[-2, :, :]   # [B, H]
            bwd_last = hidden[-1, :, :]   # [B, H]
            h_cat = torch.cat((fwd_last, bwd_last), dim=1)  # [B, 2H]
            h = h_cat
        else:
            # Última capa
            h = hidden[-1, :, :]  # [B, H]

        h = self.dropout(h)
        logits = self.fc(h)       # [B, num_classes]
        return logits


# ---------------------- TRAIN / EVAL ----------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    use_clipping: bool = False, max_norm: float = 1.0):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        # Gradient clipping opcional
        if use_clipping:
            clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch_y.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    # devolvemos ambos: F1 como métrica principal
    return avg_loss, f1, acc, all_labels, all_preds


# ---------------------- PLOTS & GUARDADO ----------------------
def plot_curves(train_losses, val_losses, val_f1s):
    epochs = range(1, len(train_losses) + 1)

    # F1 curve
    plt.figure()
    plt.plot(epochs, val_f1s, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1 (macro)")
    plt.title("Validation F1 (macro) per Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "f1_curve.png"))
    plt.close()

    # Loss curve (solo train)
    plt.figure()
    plt.plot(epochs, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss per Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
    plt.close()

    # Train vs Val loss
    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "train_val_loss_curve.png"))
    plt.close()


def plot_confusion_matrix(cm, class_names):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - RNN Vanilla")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_rnn.png"))
    plt.close()


# ---------------------- MAIN ----------------------
def main():
    # Hiperparámetros básicos
    EMBED_DIM = 100
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    MIN_FREQ = 2
    MAX_LEN = 40

    # 🔧 Gradient clipping (opcional)
    USE_CLIPPING = True      # pon False si quieres desactivarlo
    CLIP_MAX_NORM = 1.0      # norma máxima del gradiente

    # 🔧 Usar pesos por clase en la loss (para desbalance)
    USE_CLASS_WEIGHTS = True

    NUM_CLASSES = 3  # negative, neutral, positive
    CLASS_NAMES = ["negative", "neutral", "positive"]

    # ---------------------- Construir vocabulario ----------------------
    print("Construyendo vocabulario...")
    word2idx, idx2word = build_vocab(df["tokens"], min_freq=MIN_FREQ)
    vocab_size = len(idx2word)

    # ---------------------- Split train/val/test ----------------------
    print("Creando splits train/val/test...")

    X = df["tokens"]
    y = df["sentiment_label"]

    # primero train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # luego train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    train_df = X_train.to_frame()
    train_df["sentiment_label"] = y_train

    val_df = X_val.to_frame()
    val_df["sentiment_label"] = y_val

    test_df = X_test.to_frame()
    test_df["sentiment_label"] = y_test

    # ---------------------- Datasets y Dataloaders ----------------------
    train_dataset = SentimentDataset(train_df, word2idx, MAX_LEN)
    val_dataset   = SentimentDataset(val_df,   word2idx, MAX_LEN)
    test_dataset  = SentimentDataset(test_df,  word2idx, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    # ---------------------- Pesos por clase (para desbalance) ----------------------
    if USE_CLASS_WEIGHTS:
        class_counts = train_df["sentiment_label"].value_counts().sort_index().values
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        class_weights = total_samples / (num_classes * class_counts)
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=DEVICE
        )
        print(f"Usando pesos por clase: {class_weights}")
        loss_weight = class_weights_tensor
    else:
        print("No se están usando pesos por clase.")
        loss_weight = None

    # ---------------------- Modelo ----------------------
    model = SentimentRNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Scheduler sobre F1 (macro)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",      # maximizamos F1
        factor=0.5,      # reducimos LR a la mitad
        patience=2,      # si 2 epochs seguidas no mejora F1
        # verbose=False
    )

    best_val_f1 = 0.0
    best_model_path = os.path.join(RESULTS_DIR, "rnn_model.pth")

    train_losses = []
    val_losses   = []
    val_f1s      = []

    # ---------------------- Early Stopping ----------------------
    PATIENCE = 5
    epochs_no_improve = 0

    # ---------------------- Entrenamiento ----------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
            use_clipping=USE_CLIPPING,
            max_norm=CLIP_MAX_NORM,
        )

        val_loss, val_f1, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1 (macro): {val_f1:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Scheduler en base a F1
        scheduler.step(val_f1)

        # Guardar mejor modelo según F1-macro en validación
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            print("  -> Nuevo mejor modelo guardado (según F1-macro).")
        else:
            epochs_no_improve += 1
            print(f"  -> Sin mejora en F1. Paciencia: {epochs_no_improve}/{PATIENCE}")

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping activado tras {PATIENCE} epochs sin mejora en F1.")
            break

    # Guardar curvas de entrenamiento
    plot_curves(train_losses, val_losses, val_f1s)

    # ---------------------- Evaluación en test ----------------------
    print("\nCargando mejor modelo y evaluando en test...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_loss, test_f1, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Test Loss: {test_loss:.4f} | Test F1 (macro): {test_f1:.4f} | Test Acc: {test_acc:.4f}")

    rep = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES)

    # ---------------------- Guardar reporte de métricas ----------------------
    metrics_path = os.path.join(RESULTS_DIR, "metrics_report.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Test metrics for RNN vanilla sentiment classifier\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test F1 (macro): {test_f1:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(rep)
    print(f"Reporte de métricas guardado en {metrics_path}")

    # ---------------------- Guardar vocabulario ----------------------
    vocab_path = os.path.join(RESULTS_DIR, "rnn_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "word2idx": word2idx,
                "idx2word": idx2word,
                "pad_idx": PAD_IDX,
                "unk_idx": UNK_IDX,
                "max_len": MAX_LEN,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Vocabulario guardado en {vocab_path}")


if __name__ == "__main__":
    main()
