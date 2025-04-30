import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# Cargar modelo
modelo = tf.keras.models.load_model(r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\models\model_landmarks.h5")

# Cargar clases
clases = [
    'a', 'b', 'c', 'd', 'e', 'f', 'i', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y'
]

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Abrir cámara
cap = cv2.VideoCapture(0)

# Variables de control
prev_time = 0
capturando = False
palabra = ""
letra_actual = ""
inicio_tiempo_letra = 0
duracion_requerida = 3  # segundos que debe mantenerse la misma letra

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    letra_detectada = None

    # Procesar mano detectada
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Extraer coordenadas
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)

            # Predecir letra
            pred = modelo.predict(np.expand_dims(landmarks, axis=0), verbose=0)
            clase_idx = np.argmax(pred)
            letra_detectada = clases[clase_idx]

    # Solo capturar si está activado
    if capturando and letra_detectada:
        if letra_detectada != letra_actual:
            letra_actual = letra_detectada
            inicio_tiempo_letra = time.time()
        else:
            if time.time() - inicio_tiempo_letra >= duracion_requerida:
                palabra += letra_actual
                letra_actual = ""
                inicio_tiempo_letra = time.time() + 1000  # Desactivar temporalmente

    # Mostrar información en pantalla
    cv2.rectangle(frame, (5, 5), (630, 200), (0, 0, 0), -1)  # Fondo para texto

    cv2.putText(frame, 'Presiona [S] para iniciar, [E] para terminar', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    estado = "CAPTURANDO" if capturando else "ESPERANDO"
    color_estado = (0, 255, 0) if capturando else (0, 0, 255)

    cv2.putText(frame, f'Estado: {estado}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

    if letra_detectada:
        cv2.putText(frame, f'Letra detectada: {letra_detectada.upper()}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.putText(frame, f'Palabra formada: {palabra.upper()}', (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    # Calcular FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {int(fps)}', (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar ventana
    cv2.imshow('Sign Language Word Builder', frame)

    # Capturar teclas
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break
    elif key == ord('s'):  # Iniciar captura
        capturando = True
        palabra = ""
        letra_actual = ""
        inicio_tiempo_letra = 0
    elif key == ord('e'):  # Finalizar captura
        capturando = False

cap.release()
cv2.destroyAllWindows()
