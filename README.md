# 🤟 Reconocimiento de Abecedario en LSC con Deep Learning

Este proyecto tiene como objetivo desarrollar un sistema que reconozca las letras del abecedario de la Lengua de Señas Colombiana (LSC), tanto estáticas como dinámicas, utilizando visión por computadora y modelos de redes neuronales.

---

## 🎯 Objetivo

Implementar una solución que permita el reconocimiento de letras en LSC a partir de la cámara del dispositivo, usando modelos entrenados con landmarks extraídos mediante MediaPipe Hands.

---

## 📁 Estructura del Proyecto

```
SignRecognitionLSC
├── main_dynamics.py/  #Script para probar el modelo directamente con opencv_mediapipe
├── main_satics.py/    #App para letras estáticas
├── scripts/             # Scripts de pre procesado y entrenamiento de modelos
├── README.md
└── .gitignore
```

> ⚠️ Las carpetas `datasets/` y `models/` están excluidas del control de versiones mediante `.gitignore` para evitar subir archivos grandes.

---

## 🧠 Modelos

- **Estáticas**: Clasificador basado en redes neuronales profundas (DNN) con entrada de landmarks de MediaPipe.
- **Dinámicas**: Modelo LSTM para reconocer secuencias temporales de movimientos.

---

## 🛠 Tecnologías utilizadas

- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- OpenCV
- MediaPipe

---

## ⚙️ Preprocesamiento

- Extracción de landmarks de la mano por cuadro usando MediaPipe.
- Normalización y agrupación por clase.
- Aumento de datos: rotaciones, traslaciones, cambios de brillo, etc.

---

## 🚀 Cómo usar

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/lsc-abecedario.git
   cd lsc-abecedario
   ```



3. Ejecuta los scripts de entrenamiento o inferencia ubicados en la carpeta principal .py`.

---
