import os
import cv2
import numpy as np
import mediapipe as mp

# Paths
path_videos = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\datasetVideos"
path_landmarks = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\datasetVideos\datasetLandmarks"

# Parámetros
N_FRAMES = 30  # Número fijo de frames por secuencia
NUM_LANDMARKS = 21*3  # 21 puntos (x,y,z)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Crear carpetas de salida si no existen
os.makedirs(path_landmarks, exist_ok=True)

for clase in os.listdir(path_videos):
    path_clase = os.path.join(path_videos, clase)
    if not os.path.isdir(path_clase):
        continue

    # Crear carpeta de salida por clase
    path_out_clase = os.path.join(path_landmarks, clase)
    os.makedirs(path_out_clase, exist_ok=True)

    for video_name in os.listdir(path_clase):
        if not video_name.lower().endswith(('.mp4', '.mov', '.avi')):
            continue

        video_path = os.path.join(path_clase, video_name)
        cap = cv2.VideoCapture(video_path)

        frames_landmarks = []
        last_valid_landmarks = np.zeros(NUM_LANDMARKS)  # Inicialmente ceros

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                last_valid_landmarks = np.array(landmarks)
                frames_landmarks.append(last_valid_landmarks)
            else:
                # Si no detecta, usar el último válido
                frames_landmarks.append(last_valid_landmarks)

        cap.release()

        # Ajustar número de frames
        frames_landmarks = np.array(frames_landmarks)

        if len(frames_landmarks) == 0:
            print(f"[⚠️] Video vacío o sin manos detectadas: {video_name}")
            continue

        if len(frames_landmarks) > N_FRAMES:
            frames_landmarks = frames_landmarks[:N_FRAMES]
        elif len(frames_landmarks) < N_FRAMES:
            # Rellenar repitiendo el último
            last_frame = frames_landmarks[-1]
            padding = np.tile(last_frame, (N_FRAMES - len(frames_landmarks), 1))
            frames_landmarks = np.vstack((frames_landmarks, padding))

        # (Opcional) Normalización extra de coordenadas
        # frames_landmarks = (frames_landmarks - np.min(frames_landmarks)) / (np.max(frames_landmarks) - np.min(frames_landmarks))

        # Guardar la secuencia
        video_base_name = os.path.splitext(video_name)[0]
        output_file = os.path.join(path_out_clase, f"{video_base_name}.npy")
        np.save(output_file, frames_landmarks)

        print(f"[✅] Procesado {clase}/{video_name}")

print("\n🏁 ¡Proceso completado!")
