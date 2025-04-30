import os
from PIL import Image, ImageEnhance
import random

def aumentar_imagen(imagen, tamaño=(224, 224)):
    # Redimensionar la imagen a un tamaño específico
    imagen = imagen.resize(tamaño, Image.Resampling.LANCZOS)

    # Aumento aleatorio de brillo
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(imagen)
        imagen = enhancer.enhance(random.uniform(0.5, 1.5))

    # Aumento aleatorio de contraste
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(imagen)
        imagen = enhancer.enhance(random.uniform(0.5, 1.5))

    # Aumento aleatorio de color (simula la saturación)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Color(imagen)  # Ajusta la saturación de color
        imagen = enhancer.enhance(random.uniform(0.5, 1.5))

    # Aumento aleatorio de flip horizontal
    if random.random() < 0.5:
        imagen = imagen.transpose(Image.FLIP_LEFT_RIGHT)

    # Aumento aleatorio de rotación
    if random.random() < 0.3:
        angulo = random.randint(-15, 15)
        imagen = imagen.rotate(angulo)

    return imagen

def procesar_y_aumentar_imagenes(carpeta_origen, tamaño=(224, 224), augment=False):
    for subdir, dirs, files in os.walk(carpeta_origen):
        for archivo in files:
            if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
                ruta_completa = os.path.join(subdir, archivo)
                
                # Cargar la imagen
                imagen = Image.open(ruta_completa)
                
                # Aplicar aumentación si es necesario
                if augment:
                    imagen = aumentar_imagen(imagen, tamaño)

                # Sobrescribir la imagen con los cambios
                imagen.save(ruta_completa)
                print(f"[✅] Imagen modificada y guardada: {ruta_completa}")

# Ejemplo de uso
if __name__ == "__main__":
    carpeta_origen = r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\datasets\test"  # Ruta donde están las imágenes originales
    procesar_y_aumentar_imagenes(carpeta_origen, tamaño=(224, 224), augment=True)











# from PIL import Image, ImageOps
# from pathlib import Path
# import os

# def redimensionar_imagenes(ruta_dataset, size=(224, 224)):
#     ruta = Path(ruta_dataset)
#     carpetas = [d for d in ruta.rglob("*") if d.is_dir()]
    
#     for carpeta in carpetas:
#         for archivo in carpeta.glob("*.jpg"):
#             try:
#                 with Image.open(archivo) as img:
#                     img = img.convert("RGB")  # Asegura formato correcto

#                     # Escala manteniendo proporciones y añade padding
#                     img_redimensionada = ImageOps.pad(img, size, color=(0, 0, 0), centering=(0.5, 0.5))

#                     img_redimensionada.save(archivo)  # Reemplaza la original
#                 print(f"[✅] Redimensionada: {archivo}")
#             except Exception as e:
#                 print(f"[❌] Error en {archivo.name}: {e}")

# if __name__ == "__main__":
#     # Apunta a dataset/estaticas o dataset/dinamicas según el caso
#     redimensionar_imagenes(r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\datasets\DATASET_VIDEOS\test_dynamics", size=(224, 224))




# import cv2
# import os
# from pathlib import Path

# # Configura cada cuántos frames extraer una imagen
# FRAME_INTERVAL = 10  # Extraer una imagen cada 10 frames

# # Carpeta raíz del dataset
# DATASET_VIDEOS = Path(r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\datasets\dinamicas")  # Cambia según tu ruta
# SALIDA = Path(r"C:\Users\15-DK1031LA\Documents\scripsts_python_2025\SignRecognitionLSC\datasets\DATASET_FRAMES")   # Donde guardarás los frames

# def extraer_frames():
#     for letra_dir in DATASET_VIDEOS.iterdir():
#         if letra_dir.is_dir():
#             letra = letra_dir.name.lower()
#             salida_letra = SALIDA / letra
#             salida_letra.mkdir(parents=True, exist_ok=True)

#             for video in letra_dir.glob("*.mp4"):
#                 cap = cv2.VideoCapture(str(video))
#                 if not cap.isOpened():
#                     print(f"[❌] No se pudo abrir: {video.name}")
#                     continue

#                 frame_count = 0
#                 frame_id = 1
#                 print(f"[🎞️] Procesando {video.name}...")

#                 while True:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break

#                     if frame_count % FRAME_INTERVAL == 0:
#                         nombre_archivo = f"{video.stem}_frame{frame_id:03d}.jpg"
#                         ruta_salida = salida_letra / nombre_archivo
#                         cv2.imwrite(str(ruta_salida), frame)
#                         frame_id += 1

#                     frame_count += 1

#                 cap.release()
#                 print(f"[✅] {frame_id - 1} imágenes extraídas de {video.name}")

# if __name__ == "__main__":
#     extraer_frames()
