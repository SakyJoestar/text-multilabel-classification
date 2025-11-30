import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import copy
from torch.nn.utils import clip_grad_norm_

# --- Carpeta de resultados ---
RESULTS_DIR = os.path.join("results", "mlp")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. Carga y Preprocesamiento de Datos ---
print("Cargando datos...")
# El dataset corresponde al "Twitter US Airline Sentiment Dataset" 
df = pd.read_csv('data/Tweets.csv')

# Selección de columnas relevantes y limpieza inicial
df = df[['airline_sentiment', 'text']]
df = df.rename(columns={'airline_sentiment': 'sentiment', 'text': 'text'})
df.dropna(inplace=True)

# Mapeo de etiquetas a valores numéricos
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment_label'] = df['sentiment'].map(sentiment_map)

# Función de limpieza de texto
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Eliminar menciones
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Eliminar URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres no alfabéticos
    text = text.translate(str.maketrans('', '', string.punctuation)) # Eliminar puntuación
    text = " ".join(text.split()) # Eliminar espacios extra
    return text

print("Limpiando y preprocesando texto...")
df['cleaned_text'] = df['text'].apply(clean_text)

# --- 2. Preparación para el Modelo ---
X = df['cleaned_text']
y = df['sentiment_label'].values

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorización TF-IDF
print("Vectorizando texto con TF-IDF...")
vectorizer = TfidfVectorizer(max_features=100000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Conversión a tensores de PyTorch
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- 3. Definición del Modelo MLP en PyTorch ---
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

# --- 4. Entrenamiento del Modelo ---
INPUT_DIM = X_train_tfidf.shape[1]
HIDDEN_DIM = 128
OUTPUT_DIM = len(sentiment_map)
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 5
MIN_DELTA = 1e-3
DROPOUT_P = 0.5
WEIGHT_DECAY = 5e-4

model = MLPClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, p_drop=DROPOUT_P)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print("Iniciando entrenamiento del Perceptrón Multicapa...")
train_acc_history = []
val_acc_history = []

train_loss_history = []
val_loss_history = []

epoch_loss_history = []

best_val_loss = float('inf')
patience_counter = 0
best_state_dict = None

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    num_batches = 0

    for i in range(0, len(X_train_tensor), BATCH_SIZE):
        inputs = X_train_tensor[i:i+BATCH_SIZE]
        labels = y_train_tensor[i:i+BATCH_SIZE]
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # acumular
        running_loss += loss.item()
        num_batches += 1

    # promedio de loss de la epoch
    avg_epoch_loss = running_loss / max(1, num_batches)
    epoch_loss_history.append(avg_epoch_loss)
    
    # Promedio de loss en entrenamiento
    avg_train_loss = running_loss / max(1, num_batches)
    train_loss_history.append(avg_train_loss)

    # --- Evaluación al final de la época ---
    model.eval()
    with torch.no_grad():
        # Accuracy en train
        train_outputs = model(X_train_tensor)
        _, train_preds = torch.max(train_outputs, 1)
        train_acc = accuracy_score(y_train, train_preds.numpy())

        # Accuracy en test/validación
        val_outputs = model(X_test_tensor)
        _, val_preds = torch.max(val_outputs, 1)
        val_acc = accuracy_score(y_test, val_preds.numpy())
        val_loss = criterion(val_outputs, y_test_tensor).item()
    
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    val_loss_history.append(val_loss)

    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}')

    # Early Stopping: guardar el mejor modelo y verificar paciencia
    if val_loss + MIN_DELTA < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Deteniendo temprano en epoch {epoch+1} por no mejorar val_loss durante {PATIENCE} épocas.")
            break

# Cargar el mejor estado del modelo si existe
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# --- 5. Evaluación del Modelo ---
print("\nEvaluando modelo...")
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    y_pred = predicted.numpy()

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nReporte de Clasificación:")
report = classification_report(y_test, y_pred, target_names=sentiment_map.keys())

# Imprimir en consola
print(f"Accuracy: {accuracy:.4f}")
print("\nReporte de Clasificación:")
print(report)

# Guardar reporte en archivo .txt
metrics_path = os.path.join(RESULTS_DIR, "metrics_report.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Reporte de Clasificación:\n")
    f.write(report)
print(f"✅ Reporte guardado en '{metrics_path}'")

# Accuracy por época
plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_acc_history)+1), train_acc_history, label='Train Accuracy')
plt.plot(range(1, len(val_acc_history)+1), val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evolución de la Accuracy por Epoch')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
acc_curve_path = os.path.join(RESULTS_DIR, 'accuracy_curve.png')
plt.savefig(acc_curve_path)
print(f"✅ Curva de accuracy guardada en '{acc_curve_path}'")

# Pérdida por época
plt.figure(figsize=(8,5))
plt.plot(range(1, len(epoch_loss_history)+1), epoch_loss_history, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (CrossEntropy)')
plt.title('Evolución del Loss por Epoch')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_curve_path = os.path.join(RESULTS_DIR, 'loss_curve.png')
plt.savefig(loss_curve_path)
print(f"✅ Curva de loss guardada en '{loss_curve_path}'")

# Train vs Validation Loss
plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='Train Loss')
plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (CrossEntropy)")
plt.title("Evolución del Loss de Entrenamiento y Validación")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
train_val_loss_path = os.path.join(RESULTS_DIR, "train_val_loss_curve.png")
plt.savefig(train_val_loss_path)
plt.show()
print(f"✅ Curva de Train/Val Loss guardada en '{train_val_loss_path}'")

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=sentiment_map.keys(),
    yticklabels=sentiment_map.keys()
)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_mlp.png')
plt.savefig(cm_path)
print(f"\nMatriz de confusión guardada en '{cm_path}'")

# Guardar el modelo entrenado
model_path = os.path.join(RESULTS_DIR, 'mlp_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en '{model_path}'")

# Guardar el vectorizador TF-IDF
vec_path = os.path.join(RESULTS_DIR, 'tfidf_vectorizer.pkl')
joblib.dump(vectorizer, vec_path)
print(f"Vectorizador guardado en '{vec_path}'")
