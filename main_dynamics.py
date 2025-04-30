import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
from collections import deque

# --- CONFIGURACIONES ---
FRAME_WINDOW = 30  # Ventana temporal para secuencias
MODEL_PATH = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\models_dinamics\mejor_modelo_letras_v2.keras"
ENCODER_PATH = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\models_dinamics\label_encoder.pkl"

# --- CARGA DEL MODELO ---
model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Variables de estabilidad
prediction_history = deque(maxlen=7)  # *** Aumentado para mejor suavizado
stable_letter = ""
CONFIDENCE_THRESHOLD = 0.75  # *** Umbral de confianza mínimo

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5  # *** Nuevo parámetro añadido
)
mp_drawing = mp.solutions.drawing_utils

# --- BUFFER DE LANDMARKS ---
landmarks_buffer = deque(maxlen=FRAME_WINDOW + 10)  # *** Margen adicional

# --- INICIO DE CÁMARA ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesamiento de frame
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detección de manos
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # *** Extracción optimizada de landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            landmarks_buffer.append(landmarks)

    # *** Lógica de predicción mejorada
    if len(landmarks_buffer) >= FRAME_WINDOW:
        # *** Ventana deslizante sin borrar el buffer
        sequence = np.array(list(landmarks_buffer)[-FRAME_WINDOW:], dtype=np.float32)
        sequence = sequence.reshape(1, FRAME_WINDOW, -1)
        
        prediction = model.predict(sequence, verbose=0)
        confidence = np.max(prediction)  # *** Calculamos confianza
        
        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_idx = np.argmax(prediction)
            predicted_letter = label_encoder.inverse_transform([predicted_idx])[0]
            prediction_history.append(predicted_letter)
            
            # *** Sistema de votación mayoritaria
            if len(prediction_history) == prediction_history.maxlen:
                counts = {}
                for letter in prediction_history:
                    counts[letter] = counts.get(letter, 0) + 1
                stable_letter = max(counts, key=counts.get)

    # *** Mejor visualización con información de confianza
    text = f"Letra: {stable_letter} ({confidence*100:.1f}%)" if stable_letter else "Esperando gesto..."
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Reconocimiento de letras (movimiento)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
