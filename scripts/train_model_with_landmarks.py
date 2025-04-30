import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# ================== CONFIGURACIÓN ==================
# Directorio base
dataset_base_path = r"C:\Users\15-DK1031LA\Downloads\borrar\dataset_combinado\landmarks"

# Definir las clases (ordenadas alfabéticamente)
classes = sorted(os.listdir(os.path.join(dataset_base_path, 'train')))
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
print("Mapeo de clases:", class_to_idx)

# ================== FUNCIÓN PARA CARGAR LOS DATOS ==================
def load_data(split):
    X = []
    y = []
    split_path = os.path.join(dataset_base_path, split)
    for class_name in os.listdir(split_path):
        class_folder = os.path.join(split_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        label = class_to_idx[class_name]
        for file in os.listdir(class_folder):
            if file.endswith('.npy'):
                file_path = os.path.join(class_folder, file)
                landmarks = np.load(file_path)
                X.append(landmarks)
                y.append(label)
    return np.array(X), np.array(y)

# ================== CARGAR TRAIN, VALIDATION Y TEST ==================
X_train, y_train = load_data('train')
X_val, y_val = load_data('val')
X_test, y_test = load_data('test')

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# ================== DEFINIR MODELO ==================
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(21, activation='softmax')  # 21 clases
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================== ENTRENAMIENTO ==================
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

# ================== EVALUACIÓN ==================
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy en test: {test_accuracy*100:.2f}%")

# ================== GUARDAR MODELO ==================
model.save(r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\models\model_landmarks.h5")
print("Modelo guardado exitosamente.")

