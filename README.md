# Desafíos — Procesamiento del Lenguaje Natural I
Especialización en Inteligencia Artificial — FIUBA

Este repositorio contiene los cuatro desafíos prácticos de la materia Procesamiento del Lenguaje Natural I. Cada notebook está escrito en Python con Keras/TensorFlow y scikit-learn, e incluye análisis, visualizaciones y conclusiones propias.

---

## Estructura del repositorio

```
desafios_nlp/
├── desafio_1.ipynb          # Vectorización + Naïve Bayes
├── desafio_2.ipynb          # Custom Word Embeddings con Gensim
├── desafio_3.ipynb          # Modelo de lenguaje por caracteres (RNN/LSTM/GRU)
├── desafio_4.ipynb          # Bot QA con Encoder-Decoder LSTM
├── data_volunteers.json     # Dataset ConvAI2 (usado en desafío 4)
├── songs_dataset/           # Dataset de canciones (usado en desafío 2)
├── songs_dataset.zip        # Versión comprimida del dataset de canciones
├── my_model.keras           # Modelo entrenado del desafío 3 (GRU)
├── labels.tsv               # Etiquetas de embeddings (desafío 2, para Projector)
├── vectors.tsv              # Vectores de embeddings (desafío 2, para Projector)
└── gloveembedding.pkl       # Ver nota abajo — no incluido en el repo
```

> Nota sobre `gloveembedding.pkl`: Este archivo (aprox. 525 MB) supera el límite de GitHub y no está incluido. El desafío 4 lo descarga automáticamente desde Google Drive al ejecutar la celda correspondiente. Ver instrucciones más abajo.

---

## Desafío 1 — Vectorización de texto y clasificación con Naïve Bayes

Notebook: `desafio_1.ipynb`

### Qué se hizo

Se trabajó con el dataset clásico 20 Newsgroups (aprox. 18 000 posts en 20 categorías temáticas), disponible directamente en `sklearn`.

- Vectorización del corpus con `TfidfVectorizer`, explorando el impacto de distintos parámetros (`max_features`, `ngram_range`, `stop_words`, etc.).
- Entrenamiento de clasificadores Multinomial Naïve Bayes y ComplementNB para la tarea de clasificación de texto por categoría.
- Evaluación con métricas de clasificación (accuracy, F1, matriz de confusión) y análisis de los tokens más relevantes por clase.
- Exploración del efecto de filtrar o no las cabeceras de los posts, y cómo eso impacta en la calidad del clasificador.

### Herramientas principales
`scikit-learn` · `TfidfVectorizer` · `MultinomialNB` · `ComplementNB` · `matplotlib`

---

## Desafío 2 — Custom Embeddings con Gensim

Notebook: `desafio_2.ipynb`

### Qué se hizo

Se entrenaron embeddings de palabras propios (Word2Vec) sobre un corpus de letras de canciones en inglés del `songs_dataset`, eligiendo como artista a Rihanna.

- Preprocesamiento del corpus: limpieza, tokenización y preparación del vocabulario.
- Entrenamiento de un modelo Word2Vec con Gensim sobre las letras de Rihanna.
- Búsqueda de palabras más similares y menos similares para términos de interés.
- Reducción de dimensionalidad con PCA/t-SNE para visualizar los embeddings en 2D.
- Exportación de los vectores a formato `.tsv` (`vectors.tsv` / `labels.tsv`) para visualización interactiva en [TensorFlow Embedding Projector](https://projector.tensorflow.org/).

### Herramientas principales
`gensim` · `Word2Vec` · `scikit-learn (PCA)` · `matplotlib` · TensorFlow Embedding Projector

---

## Desafío 3 — Modelo de lenguaje con tokenización por caracteres

Notebook: `desafio_3.ipynb`

### Qué se hizo

Se entrenó un modelo de lenguaje a nivel de carácter sobre los diálogos de Baldur's Gate 3 (Larian Studios, 2023), extraídos y parseados de los archivos del juego.

- Corpus: diálogos HTML de los 5 actos principales, descargados desde Google Drive. Los textos se encuentran dentro de tags `<span class='dialog'>`.
- Tokenización por caracteres con One-Hot Encoding para cada carácter del vocabulario.
- Dataset estructurado como problema many-to-many (entrada: tokens $x_0..x_N$, target: $x_1..x_{N+1}$).
- Tres arquitecturas comparadas:
  | Modelo | Descripción |
  |--------|-------------|
  | SimpleRNN | Rápida, pero falla con dependencias largas |
  | LSTM | Mejor manejo de contexto largo, más pesada |
  | GRU | Balance entre velocidad y capacidad |
- Callback custom de perplejidad y early stopping.
- Generación de texto con Greedy Search, Beam Search determinista y Beam Search estocástico.
- Modelo entrenado guardado como `my_model.keras`.

### Herramientas principales
`TensorFlow/Keras` · `SimpleRNN` · `LSTM` · `GRU` · `BeamSearch`

### Fuente del corpus
[Google Drive — BG3 Parsed Dialogues](https://drive.google.com/drive/folders/1GR3CjFtM3u3-V5KrJkemTsAhx6VRUShy)

---

## Desafío 4 — Bot QA con Encoder-Decoder LSTM

Notebook: `desafio_4.ipynb`

### Qué se hizo

Se construyó un bot QA conversacional usando una arquitectura Encoder-Decoder con LSTM, siguiendo el patrón del ejercicio Traductor de la Clase 6. Los datos provienen del challenge [ConvAI2](http://convai.io/data/) (conversaciones en inglés).

```
Encoder: Input → Embedding(GloVe, no entrenable) → LSTM → [state_h, state_c]
Decoder: Input → Embedding(entrenable) → LSTM(initial_state=[h, c]) → Dense(softmax)
```

Pipeline completo:
1. Preprocesamiento: tokenizers separados para encoder (preguntas) y decoder (respuestas). Padding `pre` en encoder, `post` en decoder.
2. Embeddings: GloVe Twitter 50d en el encoder (no entrenable) + embedding entrenable en el decoder.
3. Entrenamiento: 50 épocas, batch size 64, validación con 10% de los datos.
4. Inferencia: modelos separados de encoder y decoder para generación token a token (hasta `<eos>` o longitud máxima).

Hiperparámetros:
| Parámetro | Valor |
|-----------|-------|
| Unidades LSTM | 128 |
| Épocas | 50 |
| Batch size | 64 |
| Embedding dim | 50 (GloVe) |
| Validación split | 10% |

### Herramientas principales
`TensorFlow/Keras` · `LSTM` · `GloVe (Twitter 50d)` · `ConvAI2 dataset`

---

## Cómo reproducir el modelo del Desafío 4 (`gloveembedding.pkl`)

El archivo `gloveembedding.pkl` no está incluido en el repo por su tamaño (aprox. 525 MB). Al ejecutar el notebook, la celda de carga de embeddings descarga el archivo automáticamente desde Google Drive usando `gdown`.

Si por algún motivo falla la descarga automática, podés hacerlo manualmente:

```bash
pip install gdown
gdown "1KY6avD5I1eI2dxQzMkR3WExwKwRq2g94" -O gloveembedding.pkl
```

O con `curl`:
```bash
curl -L -o "gloveembedding.pkl" \
  "https://drive.google.com/u/0/uc?id=1KY6avD5I1eI2dxQzMkR3WExwKwRq2g94&export=download&confirm=t"
```

Una vez que tenés el archivo en la carpeta del notebook, podés ejecutar el notebook de inicio a fin para re-entrenar el modelo completo. El entrenamiento fue realizado con una NVIDIA RTX 3070 (GPU local), por lo que los tiempos en CPU serán significativamente mayores.

> Nota técnica: se evitó el uso de `recurrent_dropout` en las capas LSTM para habilitar aceleración CuDNN en GPU. Si corrés en Colab, asegurate de activar GPU en Runtime → Change runtime type.

---

## Requisitos generales

```bash
pip install tensorflow keras scikit-learn gensim matplotlib pandas numpy gdown
```

Todos los notebooks están pensados para correr en Google Colab o en un entorno local con Python 3.10+.

---

## Autor

Gabriela Bohorquez — CEIA, FIUBA · 2026
