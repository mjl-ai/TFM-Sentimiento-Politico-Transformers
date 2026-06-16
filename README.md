# TFM: Análisis de Sentimiento Político mediante Transformers Ligeros

Este repositorio contiene el código fuente, los experimentos y los resultados del
Trabajo de Fin de Máster (TFM) **"Comparación de DistilBERT-es, ALBETO Tiny y MiniLM
Multilingual para el análisis de sentimiento político en X de las Elecciones Generales
del Perú 2021"**.

El proyecto compara el rendimiento de **tres modelos Transformer ligeros** en la
clasificación de sentimiento en español (negativo / neutro / positivo), evaluando el
impacto del **desbalance de clases** y del **número máximo de épocas** de entrenamiento.

## 🤖 Modelos comparados

| Modelo | Checkpoint (Hugging Face) |
| --- | --- |
| DistilBERT-es | `dccuchile/distilbert-base-spanish-uncased` |
| ALBETO Tiny | `dccuchile/albert-tiny-spanish` |
| MiniLM Multilingual | `microsoft/Multilingual-MiniLM-L12-H384` |

## 📊 Datos utilizados

El código consume los datos directamente desde el repositorio oficial para garantizar la
reproducibilidad.

- **Dataset:** `data_extended.csv` (corpus extendido de Alva-Segura, 2021)
- **Origen:** [Ver archivo en GitHub](https://github.com/danielalva2008/TFM/blob/master/TFM_UNIR/DATASETS/2.-Dataset%20Trabajados/data_extended.csv)
- **URL Raw:** `https://raw.githubusercontent.com/danielalva2008/TFM/master/TFM_UNIR/DATASETS/2.-Dataset%20Trabajados/data_extended.csv`

Tras la limpieza (corrección de codificación UTF-8, eliminación de URLs, menciones,
hashtags, textos vacíos y filtrado de etiquetas válidas {0,1,2}) se aplica una
**deduplicación exacta por texto**, que reduce el corpus a aproximadamente **4.800 tuits
únicos**. Esta es la base del escenario desbalanceado; el escenario balanceado se obtiene
por *undersampling* estratificado a la clase minoritaria.

## 📂 Estructura del repositorio

- **`/notebooks`**: contiene el cuaderno `.ipynb` con el código de entrenamiento y evaluación.
  - `TransformerLigeros_caso_balanceado_y_desbalanceado.ipynb`: script principal. Incluye
    dos celdas de código autocontenidas — **Celda 1 (escenario desbalanceado)** y
    **Celda 2 (escenario balanceado con undersampling)**.
- **`/data`**: referencia a los conjuntos de datos; contiene `data_extended.csv`.
- **`Requirements.txt`**: librerías necesarias (`transformers==4.44.2`, `torch`,
  `datasets`, `evaluate`, `accelerate`, `scikit-learn`, `wordcloud`, `ftfy`, etc.).

## 🚀 Ejecución en Google Colab

Debido a que GitHub puede presentar errores de visualización con archivos `.ipynb`
pesados (metadatos de widgets), se recomienda abrir el cuaderno directamente en Google
Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mjl-ai/TFM-Sentimiento-Politico-Transformers/blob/main/notebooks/TransformerLigeros_caso_balanceado_y_desbalanceado.ipynb)

> Se recomienda ejecutar con **GPU** (`Entorno de ejecución → Cambiar tipo de entorno → GPU`).

## 🛠️ Metodología

### Variantes de texto

El entrenamiento de los modelos se realiza sobre dos representaciones del texto, que
permiten medir el efecto de la normalización ortográfica del español:

1. **`text_tr`** — normalización suave que **conserva tildes y ñ** (minúsculas, corrección
   de *mojibake*, eliminación de URLs/menciones/hashtags y normalización de jerga peruana).
2. **`text_tr_noacc`** — la misma limpieza, pero **eliminando tildes y acentos**.

Adicionalmente se genera **`text_eda`** (normalización agresiva con eliminación de
*stopwords*), utilizada **únicamente para el análisis exploratorio** (TF-IDF, χ² y nubes
de palabras), nunca para el entrenamiento.

### Diseño experimental

Por cada escenario se entrenan **3 modelos × 2 variantes de texto = 6 ejecuciones**. El
número máximo de épocas (`EPOCHS`) es configurable y en el TFM se evalúa para **4, 8 y 16**.

### Configuración de entrenamiento

- Fine-tuning supervisado con el `Trainer` de Hugging Face (optimizador **AdamW**,
  pérdida **Cross-Entropy**).
- `MAX_LEN = 128`, `batch_size = 16`, `learning_rate = 2e-5`, `weight_decay = 0.01`.
- **Early stopping** (`patience = 2`) con selección del mejor modelo en validación según
  `f1_macro` (`load_best_model_at_end=True`).
- Particiones **estratificadas** train/val/test (70/15/15) y `seed = 42` para garantizar
  reproducibilidad. Uso de `fp16` cuando hay GPU disponible.

> **Nota metodológica:** en el escenario balanceado el *undersampling* se aplica **antes**
> del split, por lo que el conjunto de test también queda balanceado. En consecuencia, la
> *Accuracy* entre escenarios no es directamente comparable; para la comparación entre
> escenarios se prioriza el **F1-macro**, robusto al desbalance.

## 📈 Resultados

Al ejecutar cada celda se genera automáticamente una carpeta de salida
(`outputs_desbalanceado/` u `outputs_balanceado/`) con:

- **Figuras** (`figures/`): distribución de clases, nubes de palabras, TF-IDF/χ², matrices
  de confusión y curvas ROC multiclase.
- **Reportes** (`reports/`): `classification_report.txt` y `test_metrics.json` por cada
  ejecución.
- **Tablas de resultados**: `resultados_<escenario>.csv` y `.md` con Accuracy, F1-macro,
  Precision-macro, Recall-macro y AUC-macro-OVR.
- Un **ZIP** con todos los artefactos del TFM (PNG, CSV, MD, TXT, JSON). No incluye los
  pesos de los modelos para evitar archivos pesados.

## ⚙️ Instalación local (opcional)

```bash
pip install -r Requirements.txt
```

El cuaderno está pensado para Colab, pero puede ejecutarse en local con GPU y las
dependencias del archivo `Requirements.txt`.

## 📄 Licencia y cita

Trabajo académico desarrollado en el marco del Máster Universitario en Inteligencia
Artificial (UNIR). El dataset original pertenece a Alva-Segura (2021); consúltese su
repositorio para las condiciones de uso correspondientes.
