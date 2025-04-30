import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ruta al dataset original con letras estáticas
DATASET_ORIGINAL = Path(r"C:\Users\15-DK1031LA\Downloads\borrar\dataset_combinado")

# Rutas destino
DATASET_DESTINO = Path(r"C:\Users\15-DK1031LA\Downloads\borrar\dataset_combinado")
RUTAS = {
    "train": DATASET_DESTINO / "train",
    "val": DATASET_DESTINO / "val",
    "test": DATASET_DESTINO / "test"
}

# Porcentaje de división
VAL_PCT = 0.15
TEST_PCT = 0.15
TRAIN_PCT = 1 - VAL_PCT - TEST_PCT

def crear_estructura_directorios(letra):
    for tipo in RUTAS:
        carpeta = RUTAS[tipo] / letra
        carpeta.mkdir(parents=True, exist_ok=True)

def mover_imagenes(imagenes, destino, letra):
    for img_path in imagenes:
        destino_path = RUTAS[destino] / letra / img_path.name
        shutil.copy2(img_path, destino_path)

def dividir_dataset():
    for letra_folder in DATASET_ORIGINAL.iterdir():
        if letra_folder.is_dir() and letra_folder.name.isalpha():
            letra = letra_folder.name.lower()
            crear_estructura_directorios(letra)

            imagenes = list(letra_folder.glob("*.jpg"))
            if not imagenes:
                continue

            # División en train, val y test
            train_imgs, temp_imgs = train_test_split(imagenes, test_size=1 - TRAIN_PCT, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=TEST_PCT / (TEST_PCT + VAL_PCT), random_state=42)

            mover_imagenes(train_imgs, "train", letra)
            mover_imagenes(val_imgs, "val", letra)
            mover_imagenes(test_imgs, "test", letra)

            print(f"[✅] Letra '{letra}': {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

if __name__ == "__main__":
    dividir_dataset()
