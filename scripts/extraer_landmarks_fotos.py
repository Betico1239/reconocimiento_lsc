import os
import cv2
import numpy as np
import mediapipe as mp

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Ruta del dataset
dataset_path = r"C:\Users\15-DK1031LA\Downloads\borrar\dataset_combinado"
output_path = os.path.join(dataset_path, 'landmarks')
os.makedirs(output_path, exist_ok=True)

# Función para procesar imagen
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error cargando imagen: {image_path}")
        return None, None

    height, width, _ = image.shape
    center_x, center_y = 0.5, 0.5  # Centro en coordenadas normalizadas (0-1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        best_hand = None
        min_distance = float('inf')

        for hand_landmarks in results.multi_hand_landmarks:
            avg_x = np.mean([lm.x for lm in hand_landmarks.landmark])
            avg_y = np.mean([lm.y for lm in hand_landmarks.landmark])

            distance = np.sqrt((avg_x - center_x) ** 2 + (avg_y - center_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                best_hand = hand_landmarks
        
        if best_hand is None:
            return None, None

        landmarks = []
        for lm in best_hand.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        return np.array(landmarks).flatten(), best_hand
    else:
        return None, None

# === CONFIGURACIÓN DE REANUDACIÓN ===
start_split = 'train'    # 'train', 'val' o 'test'
start_class = 'q'        # nombre de la carpeta de la clase desde donde reanudar
start_file = ''          # si quieres reanudar desde un archivo específico, pon su nombre. Sino, deja ''.

start_reached = False

# Procesar todo
splits = ['train', 'val', 'test']

for split in splits:
    split_folder = os.path.join(dataset_path, split)
    for label_folder in os.listdir(split_folder):
        label_folder_path = os.path.join(split_folder, label_folder)
        if not os.path.isdir(label_folder_path):
            continue
        
        # Saltar hasta que lleguemos a start_split y start_class
        if not start_reached:
            if split == start_split and label_folder == start_class:
                start_reached = True
            else:
                continue

        # Crear carpeta de salida
        output_split_label = os.path.join(output_path, split, label_folder)
        os.makedirs(output_split_label, exist_ok=True)

        for image_file in os.listdir(label_folder_path):
            if not start_reached:
                continue

            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Si definiste un archivo de inicio específico
                if start_file:
                    if image_file < start_file:
                        continue
                landmarks, hand_landmarks_obj = extract_landmarks(os.path.join(label_folder_path, image_file))

                if landmarks is not None:
                    # Guardar landmarks
                    filename = os.path.splitext(image_file)[0] + '.npy'
                    save_path = os.path.join(output_split_label, filename)
                    np.save(save_path, landmarks)

                    # Dibujar y mostrar
                    image_bgr = cv2.imread(os.path.join(label_folder_path, image_file))
                    mp_drawing.draw_landmarks(image_bgr, hand_landmarks_obj, mp_hands.HAND_CONNECTIONS)
                    cv2.imshow('Hand Landmarks', image_bgr)
                    key = cv2.waitKey(500)
                    if key == 27:  # ESC
                        cv2.destroyAllWindows()
                        exit()
                else:
                    print(f"[ADVERTENCIA] No se detectaron manos en {split}/{label_folder}/{image_file}")

cv2.destroyAllWindows()
print("\nProceso completado.")
