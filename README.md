# Clasificación de Mamografías con Modelos Multimodales

Este proyecto fue desarrollado como trabajo final del curso **Visión por Computador** de la **Maestría en Ciencia de los Datos y Analítica** de la **Universidad EAFIT**. El objetivo principal fue construir un modelo de deep learning que combine datos **visuales** (mamografías) y **tabulares** (características clínicas) para realizar una clasificación precisa tratando de emular las decisiones que toma un radiologo al momento de decidir si un paciente debe ser remitido a estudios mas avanzados y determinar si este tiene o no cáncer de seno.

Proyecto basado en la información de la competencia en kaggle [text](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)

---

## Objetivo

Desarrollar un modelo de aprendizaje profundo multimodal capaz de predecir categorías clínicas asociadas a mamografías, integrando imágenes y datos estructurados para mejorar la capacidad diagnóstica.

---

## Curso

- **Materia**: Visión por Computador  
- **Maestría**: Ciencia de los Datos y Analítica  
- **Universidad**: EAFIT  
- **Año**: 2025

---

## Autores

- Fabian David Sánchez Martínez
- Mateo Holguín Carvalho
- Melissa Andrea Ciro Peña

---

## Tecnologías utilizadas

- Python 3.10
- TensorFlow / Keras
- Pandas / NumPy
- OpenCV
- scikit-learn
- Matplotlib / Seaborn
- Streamlit

---

## Estructura del Proyecto

```plaintext
📦 cancer_classification/
│
├── 📁 data/                   # Datos tabulares e imágenes de mamografías, solo se carga una muestra de imagenes para ver las imagenes totales ver: [text](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
├── 📁 notebooks/              # Exploración, preprocesamiento y entrenamiento
├── 📁 streamlit_app/          # Aplicación para despliegue interactivo
├── 📄 requirements.txt        # Dependencias del proyecto
├── 📄 README.md               # Documentación del proyecto

```

---

## ¿Cómo ejecutar el proyecto?

### 1. Clona el repositorio

```bash
git clone https://github.com/usuario/repositorio.git
cd repositorio
```

### 2. Crea un entorno virtual e instala las dependencias

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ejecuta la aplicación con Streamlit

```bash
cd streamlit_app
streamlit run cancer_app_streamlit.py
```

### 4. O bien, explora los notebooks

```bash
cd notebooks
# Abre los notebooks para ver exploración de datos, entrenamiento y resultados
```

---

## Resultados

Precisión promedio del modelo multimodal: 97%
Modelos utilizados: ResNet18, VGG16, combinación densa final
Técnicas de interpretación: GradCAM
Métricas evaluadas: Accuracy, Recall y F1-Score por clase

---

## Enfoque Multimodal

El modelo combina dos fuentes de información:

- Imágenes de mamografías procesadas con CNNs (ResNet18)
- Datos tabulares clínicos procesados con una red neuronal multicapa
-Ambas salidas se concatenan para una predicción conjunta en una capa densa final.

---

## Referencias
"Deep Learning to Improve Breast Cancer Detection on Screening Mammography"
Li Shen, Laurie R. Margolies, Joseph H. Rothstein, Eugene Fluder, Russell McBride & Weiva Sieh
Scientific Reports, 9, Article number: 12495 (2019)

---

## Licencia
Este proyecto fue desarrollado con fines académicos. No debe utilizarse en entornos clínicos reales sin la debida validación y aprobación ética.

Universidad EAFIT – Maestría en Ciencia de los Datos y Analítica
Curso: Visión por Computador – 2025
