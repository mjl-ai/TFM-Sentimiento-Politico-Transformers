# TFM: Análisis de Sentimiento Político mediante Transformers Ligeros

Este repositorio contiene el código fuente, los experimentos y los resultados del Trabajo de Fin de Máster (TFM). El proyecto se centra en la comparación del rendimiento de modelos **DistilBERT** bajo escenarios de datos balanceados y desbalanceados, utilizando diferentes niveles de preprocesamiento.

## 📊 Datos Utilizados
El modelo consume los datos directamente desde el repositorio oficial para garantizar la reproducibilidad.
* **Dataset:** `data_extended.csv`
* **Origen:** [Ver archivo en GitHub](https://github.com/danielalva2008/TFM/blob/master/TFM_UNIR/DATASETS/2.-Dataset%20Trabajados/data_extended.csv)
* **URL Raw:** `https://raw.githubusercontent.com/danielalva2008/TFM/master/TFM_UNIR/DATASETS/2.-Dataset%20Trabajados/data_extended.csv`

## 📂 Estructura del Repositorio
* **/notebooks**: Contiene los archivos `.ipynb` con el código de entrenamiento y evaluación.
  * `TransformerLigeros_Balanceado_y_Desbalanceado.ipynb`: Script principal que ejecuta los 6 experimentos (3 niveles de texto x 2 tipos de balanceo).
* **/data**: Referencia a los conjuntos de datos, contiene el dataset data_extended.csv.
* `requirements.txt`: Listado de librerías necesarias (`transformers`, `torch`, `evaluate`, etc.).

## 🚀 Ejecución en Google Colab
Debido a que GitHub puede presentar errores de visualización con archivos `.ipynb` pesados (metadatos de widgets), se recomienda abrir el código directamente en Google Colab a través del siguiente botón:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielalva2008/TFM/blob/main/notebooks/TransformerLigeros_Balanceado_y_Desbalanceado.ipynb)

*(Nota: Asegúrate de que la ruta en el enlace superior coincida exactamente con el nombre de tu archivo en GitHub).*

## 🛠️ Metodología
Se entrenaron modelos DistilBERT evaluando tres variantes del texto:
1. **Raw UTF-8**: Solo corrección de caracteres rotos.
2. **Fixed**: Limpieza de URLs, menciones y espacios.
3. **NoAcc**: Limpieza "Fixed" eliminando además tildes y acentos.

Se compararon métricas de **F1-Score (Macro)**, **Accuracy** y **AUC** para determinar el impacto del balanceo de clases y la normalización del lenguaje coloquial.

## 📈 Resultados
Los resultados detallados (matrices de confusión, nubes de palabras y reportes de clasificación) se generan automáticamente en una carpeta llamada `outputs_tfm/` al ejecutar el código.
