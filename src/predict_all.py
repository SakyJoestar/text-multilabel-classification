import torch
import torch.nn as nn
import joblib
import re
import string
import json

# Para reutilizar el mismo preprocesamiento de la RNN/GRU
from scripts.preprocessing import preprocess_text as rnn_preprocess_text

# ---------------- CONFIGURACIÓN GENERAL ----------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {0: "negative", 1: "neutral", 2: "positive"}

# ---------------- MLP: MODELO + VECTORIZADOR ----------------

# Cargar el vectorizador primero para obtener INPUT_DIM dinámicamente
VEC_PATH = "results/mlp/tfidf_vectorizer.pkl"
MLP_MODEL_PATH = "results/mlp/mlp_model.pth"

vectorizer = joblib.load(VEC_PATH)
INPUT_DIM = len(vectorizer.get_feature_names_out())
HIDDEN_DIM_MLP = 128
OUTPUT_DIM = 3
DROPOUT_P = 0.5


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, p_drop=0.3):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out


mlp_model = MLPClassifier(INPUT_DIM, HIDDEN_DIM_MLP, OUTPUT_DIM, p_drop=DROPOUT_P)
mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=DEVICE))
mlp_model.to(DEVICE)
mlp_model.eval()

# --- limpieza específica del MLP (la misma que en el entrenamiento) ---
def clean_text_mlp(text: str) -> str:
    text = text.lower()
    text = re.sub(r"@[a-zA-Z0-9_]+", "", text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


# ---------------- RNN / GRU: MODELOS + VOCABULARIOS ----------------

# Arquitectura usada en train_rnn / train_gru (ajusta si cambiaste allá)
EMBED_DIM = 200
HIDDEN_DIM_RNN = 256
NUM_LAYERS_RNN = 2
BIDIRECTIONAL = True
DROPOUT_RNN = 0.4
NUM_CLASSES = 3


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
            padding_idx=0,  # asumimos PAD_IDX = 0
        )

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        embedded = self.embedding(x)          # [B, L, E]
        output, hidden = self.rnn(embedded)   # hidden: [num_layers * num_dir, B, H]

        if self.bidirectional:
            fwd_last = hidden[-2, :, :]
            bwd_last = hidden[-1, :, :]
            h = torch.cat((fwd_last, bwd_last), dim=1)
        else:
            h = hidden[-1, :, :]

        h = self.dropout(h)
        logits = self.fc(h)
        return logits


class SentimentGRU(nn.Module):
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
        super(SentimentGRU, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,  # asumimos PAD_IDX = 0
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        embedded = self.embedding(x)          # [B, L, E]
        output, hidden = self.gru(embedded)   # hidden: [num_layers * num_dir, B, H]

        if self.bidirectional:
            fwd_last = hidden[-2, :, :]
            bwd_last = hidden[-1, :, :]
            h = torch.cat((fwd_last, bwd_last), dim=1)
        else:
            h = hidden[-1, :, :]

        h = self.dropout(h)
        logits = self.fc(h)
        return logits


# ---- utilidades para RNN/GRU ----

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    word2idx = data["word2idx"]
    max_len = data["max_len"]
    pad_idx = data["pad_idx"]
    unk_idx = data["unk_idx"]
    return word2idx, max_len, pad_idx, unk_idx


def tokens_to_ids(tokens, word2idx, unk_idx: int):
    return [word2idx.get(tok, unk_idx) for tok in tokens]


def pad_ids(ids, max_len: int, pad_idx: int):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        return ids + [pad_idx] * (max_len - len(ids))


# Cargar artefactos RNN
RNN_MODEL_PATH = "results/RNN/rnn_model.pth"
RNN_VOCAB_PATH = "results/RNN/rnn_vocab.json"

word2idx_rnn, MAX_LEN_RNN, PAD_IDX_RNN, UNK_IDX_RNN = load_vocab(RNN_VOCAB_PATH)
vocab_size_rnn = len(word2idx_rnn)

rnn_model = SentimentRNN(
    vocab_size=vocab_size_rnn,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM_RNN,
    num_classes=NUM_CLASSES,
    num_layers=NUM_LAYERS_RNN,
    bidirectional=BIDIRECTIONAL,
    dropout=DROPOUT_RNN,
)
rnn_model.load_state_dict(torch.load(RNN_MODEL_PATH, map_location=DEVICE))
rnn_model.to(DEVICE)
rnn_model.eval()

# Cargar artefactos GRU (ajusta paths según tu entrenamiento)
GRU_MODEL_PATH = "results/GRU/gru_model.pth"
GRU_VOCAB_PATH = "results/GRU/gru_vocab.json"

try:
    word2idx_gru, MAX_LEN_GRU, PAD_IDX_GRU, UNK_IDX_GRU = load_vocab(GRU_VOCAB_PATH)
    vocab_size_gru = len(word2idx_gru)

    gru_model = SentimentGRU(
        vocab_size=vocab_size_gru,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM_RNN,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS_RNN,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT_RNN,
    )
    gru_model.load_state_dict(torch.load(GRU_MODEL_PATH, map_location=DEVICE))
    gru_model.to(DEVICE)
    gru_model.eval()
    GRU_AVAILABLE = True
except FileNotFoundError:
    gru_model = None
    GRU_AVAILABLE = False
    print("⚠️ Modelo GRU no encontrado. Solo estarán disponibles MLP y RNN.")


# ---------------- FUNCIÓN DE PREDICCIÓN UNIFICADA ----------------

def predict_sentiment(text: str, model_type: str = "mlp") -> str:
    """
    Realiza una predicción de sentimiento usando el modelo indicado.
    
    model_type: "mlp", "rnn" o "gru"
    """
    model_type = model_type.lower()

    if model_type == "mlp":
        # 1. Limpieza tipo MLP
        cleaned_text = clean_text_mlp(text)
        # 2. TF-IDF
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()
        text_tensor = torch.tensor(vectorized_text, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            output = mlp_model(text_tensor)
            _, predicted_idx = torch.max(output, 1)

        predicted_label = label_map[predicted_idx.item()]
        return predicted_label

    elif model_type in ("rnn", "gru"):
        # 1. Preprocesar texto con el pipeline de la RNN/GRU
        # (limpieza + tokenización + normalización)
        tokens = rnn_preprocess_text(text)

        if model_type == "rnn":
            word2idx = word2idx_rnn
            max_len = MAX_LEN_RNN
            pad_idx = PAD_IDX_RNN
            unk_idx = UNK_IDX_RNN
            model = rnn_model
        else:
            if not GRU_AVAILABLE:
                raise RuntimeError("El modelo GRU no está disponible (no se encontraron archivos).")
            word2idx = word2idx_gru
            max_len = MAX_LEN_GRU
            pad_idx = PAD_IDX_GRU
            unk_idx = UNK_IDX_GRU
            model = gru_model

        # 2. Convertir a ids + padding
        ids = tokens_to_ids(tokens, word2idx, unk_idx)
        ids = pad_ids(ids, max_len, pad_idx)

        seq_tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            output = model(seq_tensor)
            _, predicted_idx = torch.max(output, 1)

        predicted_label = label_map[predicted_idx.item()]
        return predicted_label

    else:
        raise ValueError(f"Modelo no soportado: {model_type}. Usa 'mlp', 'rnn' o 'gru'.")


# ---------------- EJEMPLO DE USO ----------------
if __name__ == "__main__":
    import random

    negative_tweets = [
        "@AmericanAir your customer service is the worst. I've been on hold for 3 hours.",
        "@united Lost my baggage again. I'm beyond frustrated with your airline.",
        "My flight was delayed 5 hours with no explanation or apology. Terrible experience.",
        "The crew was rude and unhelpful the entire trip. Never flying with you again.",
        "Seats were filthy and the cabin smelled awful. Completely unacceptable.",
        "You cancelled my flight last minute and offered no real compensation. I'm furious.",
        "Customer support keeps sending automated replies and no real solution. So disappointing.",
        "The check-in system crashed and I almost missed my flight. Absolute chaos.",
        "@SouthwestAir you overbooked my flight and kicked me off. I'm done with this airline.",
        "Charging extra for everything and still giving awful service. What a joke.",
        "The pilot kept us on the runway for an hour without updates. Miserable.",
        "Lost luggage, broken stroller, and zero responsibility taken. I'm exhausted and angry.",
        "Your app is useless. It keeps crashing while I'm trying to check in.",
        "Missed my connection because your staff gave me the wrong gate. Unbelievable.",
        "Cabin temperature was freezing and nobody cared when we complained.",
        "The food was inedible and made me feel sick. Disgusting.",
        "You changed my seat without notice and separated me from my family.",
        "@delta Worst boarding process I've ever seen. No organization at all.",
        "Flight attendants ignored call buttons the whole flight. Terrible attitude.",
        "Delayed, overbooked, and no apology. This airline keeps getting worse.",
    ]

    neutral_tweets = [
        "Flight departed at 9:05 as scheduled.",
        "Checked in online without any issues.",
        "The legroom was average, nothing special.",
        "Boarding process was acceptable, just a bit slow.",
        "Got a middle seat but the flight was on time.",
        "The cabin was a little noisy but overall fine.",
        "Security lines were long, but that's to be expected.",
        "Snacks were the usual chips and cookies selection.",
        "Seats were okay for a short flight.",
        "Transfer at the connecting airport went smoothly.",
        "Gate agents provided basic information when asked.",
        "Entertainment options were limited but worked.",
        "The plane landed a few minutes later than expected.",
        "My luggage arrived with some minor scuffs.",
        "Weather caused a small delay, but nothing major.",
        "Check-in kiosk worked fine for printing my boarding pass.",
        "Standard economy seat, average comfort level.",
        "The announcement volume was a bit low but understandable.",
        "Not the best flight, not the worst either.",
        "Average experience overall—nothing to complain about, nothing standout.",
    ]

    positive_tweets = [
        "@united Thanks for the great flight and friendly staff! Really enjoyed it.",
        "Smooth boarding, comfy seats, and we arrived earlier than expected. Loved it.",
        "The crew was incredibly kind and attentive the whole time.",
        "Fantastic in-flight entertainment and super fast Wi-Fi.",
        "Check-in was quick and easy, staff were smiling and helpful.",
        "Best legroom I've had in economy, very comfortable flight.",
        "The pilot gave clear updates and a very smooth landing.",
        "Free snacks and drinks made the short flight even better.",
        "@SouthwestAir thanks for rebooking me so quickly after my delay.",
        "I got a complimentary upgrade and the service was outstanding.",
        "Cabin was clean, quiet, and the lighting was really relaxing.",
        "The app worked perfectly for check-in and choosing my seat.",
        "Friendly gate agents made the boarding process stress-free.",
        "Really appreciated how the crew helped families with small kids.",
        "The food was surprisingly tasty and nicely presented.",
        "My luggage came out first on the carousel, super fast.",
        "Seats were comfortable and the blankets were soft and clean.",
        "Staff handled a minor issue on board very professionally.",
        "I felt safe, cared for, and welcome during the entire flight.",
        "Overall an excellent experience—I'll happily fly with you again.",
    ]

# Unimos y generamos 15 tweets aleatorios de cualquier etiqueta
    all_tweets = negative_tweets + neutral_tweets + positive_tweets
    sample_tweets = random.sample(all_tweets, 15)

    for i, tw in enumerate(sample_tweets, start=1):
        print(f"Tweet {i}: '{tw}'")
        print(f"  MLP  -> {predict_sentiment(tw, model_type='mlp')}")
        print(f"  RNN  -> {predict_sentiment(tw, model_type='rnn')}")
        if GRU_AVAILABLE:
            print(f"  GRU  -> {predict_sentiment(tw, model_type='gru')}")
        print()
