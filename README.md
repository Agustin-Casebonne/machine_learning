# 🤖 Prácticas de Clasificación con Machine Learning (Scikit-learn)

Este repositorio contiene dos notebooks de Jupyter que exploran diversas técnicas de **Clasificación** en Machine Learning utilizando Python y la librería `scikit-learn` entre.

---

## 🌸 1. `Practica_Iris_Clasificacion.ipynb` (Clasificadores Base y SVM)

Análisis centrado en la clasificación de las especies de flores del dataset **Iris**, comparando modelos básicos y explorando el impacto de los hiperparámetros.

### ⚙️ Metodología

1.  **Preprocesamiento:** Se realizó la codificación de la variable objetivo (`LabelEncoder`) y la normalización de las características (`MinMaxScaler`).
2.  **División de Datos:** Entrenamiento (80%) y Prueba (20%) con `random_state=42`.

### 🎯 Hallazgos Clave

| Modelo | Exactitud | Comentario |
| :--- | :--- | :--- |
| **SVM Lineal** | **1.0000** | Máximo desempeño, indicando alta separabilidad. |
| **KNN (k=5)** | **1.0000** | Rendimiento perfecto y robusto. |
| Regresión Logística | 0.9667 | Excelente desempeño, cerca del máximo. |
| Perceptrón | 0.8333 | Desempeño aceptable, pero menor que el resto. |

#### Influencia del Hiperparámetro C (SVM Lineal)

Al reducir el valor de **C**, el modelo se vuelve más tolerante a errores, lo que resulta en un **margen más suave** y un **mayor número de Vectores Soporte**.

| Valor de C | Vectores Soporte | Efecto en el Margen |
| :--- | :--- | :--- |
| **0.1** | 66 | Más suave (mayor generalización). |
| **10** | 7 | Más estricto (menor generalización). |

---

## 🚀 2. `Practica_con_ensambles.ipynb` (Ensambles y Optimización)

Exploración de modelos más complejos, enfocándose en la optimización de hiperparámetros (`GridSearchCV`) y el uso de técnicas de **Ensambles** (`StackingClassifier`) en diferentes datasets de clasificación multiclase y binaria.

### 📂 Datasets Analizados

1.  **EPL (Fútbol):** Predicción de resultado (`H`, `A`, `D`).
2.  **Fallo Cardíaco:** Predicción de supervivencia (`DEATH_EVENT`).
3.  **Semillas:** Predicción del tipo de semilla (multiclase).

### 📈 Rendimiento de Modelos (Resumen)

| Dataset | Tarea | Mejor Exactitud | Mejor Modelo / Técnica |
| :--- | :--- | :--- | :--- |
| **EPL** | Multiclase | 99.95% | SVC (Posible *Data Leakage*) |
| **Fallo Cardíaco** | Binaria | **83.33%** | **StackingClassifier** |
| **Semillas** | Multiclase | **95.24%** | **SVC Optimizado** |

### 🛠️ StackingClassifier (Fallo Cardíaco)

El modelo de ensamble demostró la **mejor capacidad de generalización** para el problema de fallo cardíaco.

* **Estimadores Base:** `GaussianNB`, `SVC`, `DecisionTreeClassifier`.
* **Estimador Final:** `LogisticRegression`.
* **Exactitud (General):** 83.33%

### 💡 Lecciones Aprendidas

* **Preprocesamiento:** La normalización es fundamental, especialmente para **SVM** y **KNN**, como se demostró en el dataset de Semillas.
* **Optimización:** La búsqueda de hiperparámetros (`GridSearchCV`) permitió que un **SVC Polinómico** lograra la mejor exactitud en el dataset de Semillas (95.24%), superando a los ensambles.
* **Métricas de Clasificación:** Es vital usar métricas completas como el **Reporte de Clasificación** para entender el desempeño (e.g., el modelo `GaussianNB` en Fallo Cardíaco tuvo una alta Precisión pero baja Sensibilidad).