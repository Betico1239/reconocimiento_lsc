
#PREDECIR PROBAR EL CONJUNTO TEST DE LOS LANDMARKS ESTATICAS
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el modelo
modelo = tf.keras.models.load_model(r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\models\model_landmarks.h5")

# Path donde guardaste landmarks
test_path = r"C:\Users\15-DK1031LA\Downloads\borrar\dataset_combinado\landmarks\test"

# Mapeo de clases (en orden alfabético de las carpetas)
clases = sorted(os.listdir(test_path))  # asumiendo que son carpetas a, b, c, ..., u
clase_to_idx = {clase: idx for idx, clase in enumerate(clases)}
idx_to_clase = {idx: clase for clase, idx in clase_to_idx.items()}

X_test = []
y_test = []

# Cargar todos los archivos .npy del test
for clase in clases:
    clase_folder = os.path.join(test_path, clase)
    for archivo in os.listdir(clase_folder):
        if archivo.endswith('.npy'):
            path_archivo = os.path.join(clase_folder, archivo)
            landmarks = np.load(path_archivo)
            X_test.append(landmarks)
            y_test.append(clase_to_idx[clase])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Hacer predicciones
y_pred = modelo.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Resultados
print(classification_report(y_test, y_pred_classes, target_names=clases))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de confusión en Test')
plt.show()
