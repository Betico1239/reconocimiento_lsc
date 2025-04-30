import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURACIONES ---
MODEL_PATH = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\models_dinamics\mejor_modelo_letras_v2.keras"
X_TEST_PATH = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\models_dinamics\X_test.npy"
Y_TEST_PATH = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\models_dinamics\y_test.npy"
ENCODER_PATH = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\models_dinamics\label_encoder.pkl"

# --- CARGAR DATOS Y MODELO ---
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)
label_encoder = joblib.load(ENCODER_PATH)
model = load_model(MODEL_PATH)

# --- AJUSTAR FORMA SI NECESARIO ---
# El modelo espera datos con shape (samples, N_FRAMES, NUM_LANDMARKS)
N_FRAMES = 30
NUM_LANDMARKS = 63

try:
    X_test = X_test.reshape((-1, N_FRAMES, NUM_LANDMARKS))
except:
    raise ValueError(f"❌ Las dimensiones de X_test no coinciden con (N_FRAMES={N_FRAMES}, NUM_LANDMARKS={NUM_LANDMARKS}).")

# --- PREDICCIONES ---
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# --- RESULTADOS ---
labels = label_encoder.classes_
print("\n--- Clasificación ---")
print(classification_report(y_true, y_pred, target_names=labels))

acc = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy del modelo en test: {acc:.4f}")

# --- MATRIZ DE CONFUSIÓN ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()
