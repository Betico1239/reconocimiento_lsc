# Importaciones
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# -------------------------
# 1. Configuración Inicial
# -------------------------

path_landmarks = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\datasetVideos\datasetLandmarks"

N_FRAMES = 30
NUM_LANDMARKS = 63
TEST_SIZE = 0.2
BATCH_SIZE = 8
EPOCHS = 70

# -------------------------
# 2. Carga de Datos
# -------------------------

label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(path_landmarks)))}
print("Mapping de clases:", label_map)

X = []
y = []

for label in label_map.keys():
    path_clase = os.path.join(path_landmarks, label)
    for file_name in os.listdir(path_clase):
        if file_name.endswith('.npy'):
            data = np.load(os.path.join(path_clase, file_name))
            X.append(data)
            y.append(label)

X = np.array(X)

# Codificar labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# -------------------------
# 3. Aumentación de Datos (opcional)
# -------------------------

def add_noise(data, noise_level=0.015):
    return data + np.random.normal(0, noise_level, data.shape)

X_aug = add_noise(X)
y_aug = y_categorical.copy()

X_total = np.concatenate([X, X_aug])
y_total = np.concatenate([y_categorical, y_aug])

# -------------------------
# 4. División y Guardado
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_total, y_total, test_size=TEST_SIZE, random_state=42, stratify=y_total
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# Guardar datos para uso futuro
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Guardar encoder y mapping
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(label_map, 'label_map.pkl')

# -------------------------
# 5. Modelo
# -------------------------

model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(N_FRAMES, NUM_LANDMARKS)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(LSTM(64, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(len(label_map), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# 6. Callbacks
# -------------------------

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

checkpoint = ModelCheckpoint(
    filepath='mejor_modelo_letras.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

callbacks_list = [early_stop, checkpoint, reduce_lr]

# -------------------------
# 7. Entrenamiento
# -------------------------

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list
)

# -------------------------
# 8. Guardar Modelo Final
# -------------------------

model.save('modelo_letras_movimiento_final.keras')
print("✅ Modelo entrenado y datos guardados exitosamente.")
