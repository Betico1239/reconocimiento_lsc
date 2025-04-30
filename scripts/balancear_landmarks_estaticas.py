# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 21:51:28 2025

@author: 15-DK1031LA
"""

##Balanceo de la cantidad de.npy por cada clase del dataset para entrenamiento del modelo
import os
import shutil
import random
import numpy as np

# Ruta de los landmarks generados
landmarks_path = r"C:\Users\15-DK1031LA\Downloads\borrar\dataset_combinado\landmarks"

# Configuración de perturbación
NOISE_FACTOR = 0.01  # 1% de perturbación máxima

# Procesar por split
splits = ['train', 'val', 'test']

for split in splits:
    split_folder = os.path.join(landmarks_path, split)
    print(f"\nProcesando split: {split}")
    
    # Contar archivos en cada clase
    class_counts = {}
    for class_name in os.listdir(split_folder):
        class_path = os.path.join(split_folder, class_name)
        if os.path.isdir(class_path):
            npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
            class_counts[class_name] = len(npy_files)
    
    # Determinar máximo número de ejemplos
    max_samples = max(class_counts.values())
    print(f"Máximo número de ejemplos en una clase: {max_samples}")

    # Duplicar donde sea necesario
    for class_name, count in class_counts.items():
        class_path = os.path.join(split_folder, class_name)
        npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]

        if count < max_samples:
            needed = max_samples - count
            print(f" - Aumentando {needed} ejemplos en la clase '{class_name}'")

            for i in range(needed):
                file_to_duplicate = random.choice(npy_files)
                src_file = os.path.join(class_path, file_to_duplicate)
                
                # Cargar los landmarks originales
                landmarks = np.load(src_file)

                # Aplicar ruido pequeño
                noise = np.random.uniform(-NOISE_FACTOR, NOISE_FACTOR, landmarks.shape)
                landmarks_noisy = landmarks + noise

                # Crear nombre nuevo (sin errores en entrenamiento)
                base_name = os.path.splitext(file_to_duplicate)[0]
                new_file_name = f"{base_name}_aug_{i}.npy"
                dst_file = os.path.join(class_path, new_file_name)

                # Guardar el landmarks modificado
                np.save(dst_file, landmarks_noisy)

print("\n✅ Balanceo completado con duplicación + ruido añadido.")

