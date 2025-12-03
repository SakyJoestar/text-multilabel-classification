# 🧠 Clasificación de Sentimientos en Tweets de Aerolíneas

## Introducción

Este proyecto busca analizar el desempeño de diferentes arquitecturas de redes neuronales vistas en el curso y seleccionar la más adecuada para resolver un problema de **clasificación de texto con múltiples etiquetas** usando el Framework **Pytorch**.

## Dataset

El proyecto utiliza el dataset [Twitter Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) de Kaggle.

## Entrega 1

Se implementa y evalúa un modelo de **Perceptrón Multicapa (MLP)**.

### Arquitectura del modelo

- **Tipo:** Perceptrón Multicapa (MLP)
- **Capas:**
  - Capa densa oculta de **128** neuronas
  - **ReLU** como activación en la capa oculta
  - Capa de salida con **3** neuronas ( `negative`, `neutral`, `positive`)
- **Función de activación (salida):** `Softmax`
- **Función de pérdida:** `CrossEntropyLoss`
- **Optimizador:** `Adam` con `lr = 0.001`
- **Batch size:** `64`
- **Épocas:** `10`

<!-- ## Código
(Incluir snippet del modelo en Keras)

## Resultados
(Tablas/gráficas de accuracy, precisión, recall, F1)

## Conclusiones
(Breve análisis de resultados y próximos pasos) -->

## 📥 Descarga del dataset

Para descargarlo de forma automática, sigue estos pasos:

1. **Instala la API de Kaggle**

   ```bash
   pip install kaggle
   ```

2. **Configura tu token de Kaggle**

   - Ve a tu cuenta de Kaggle → **Account** → **Create API Token**.
   - Se descargará un archivo llamado `kaggle.json`.
   - Guárdalo en la ruta correspondiente según tu sistema operativo:

     - **Linux/Mac**

       ```bash
       ~/.kaggle/kaggle.json
       ```

     - **Windows**
       ```bash
       C:\Users\<tu_usuario>\.kaggle\kaggle.json
       ```

3. **Ejecuta el script de descarga**

   En la raíz del proyecto, corre el siguiente comando:

   ```bash
   python scripts/download_dataset.py
   ```

# Entrega 1

## Ejecutar el entrenamiento para el modelo MLP

Desde el directorio raiz siga estos pasos:

1. **Instalar los requerimientos**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el entrenamiento**

   ```bash
   python src/train_mlp.py
   ```

3. **Ejecutar la predicción**

   ```bash
   python src/predict.py
   ```


4. **Observar los resultados**

   En la carpeta results se puede observar la matriz de confusión.

# Entrega 2

## Ejecutar el entrenamiento para el modelo RNN
   ```bash
   python src/train_RNN.py
   ```

## Ejecutar el entrenamiento para el modelo GRU
   ```bash
   python src/train_GRU.py
   ```

## Ejecutar el entrenamiento para el transformer
   ```bash
   python src/train_transformer.py
   ```
## Predecir usando todos los modelos
```bash
python src/predict_all.py
```