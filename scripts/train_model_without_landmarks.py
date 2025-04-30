import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def cargar_datos(carpeta_origen, tamaño=(640, 480), batch_size=32):
    # Augmentación para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,          # rotación aleatoria
        width_shift_range=0.2,      # desplazamiento horizontal
        height_shift_range=0.2,     # desplazamiento vertical
        shear_range=0.15,           # corte aleatorio
        zoom_range=0.15,            # zoom aleatorio
        horizontal_flip=True,       # voltear horizontalmente
        fill_mode='nearest'         # modo de rellenado
    )

    # Solo reescalado para validación y test
    test_val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Datos de entrenamiento con augmentación
    train_data = train_datagen.flow_from_directory(
        os.path.join(carpeta_origen, 'train'),
        target_size=tamaño,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Validación
    validation_data = test_val_datagen.flow_from_directory(
        os.path.join(carpeta_origen, 'val'),
        target_size=tamaño,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Test
    test_data = test_val_datagen.flow_from_directory(
        os.path.join(carpeta_origen, 'test'),
        target_size=tamaño,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_data, validation_data, test_data

def crear_modelo(input_shape=(640, 480, 3), num_clases=21):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(num_clases, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    # Congelar capas del modelo base
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def entrenar_modelo(modelo, train_data, validation_data, epochs=20, batch_size=32):
    modelo.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    historial = modelo.fit(
        train_data,
        steps_per_epoch=train_data.samples // batch_size,
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=validation_data.samples // batch_size
    )
    
    return historial

# === Entrenamiento ===
if __name__ == "__main__":
    carpeta_origen = r"C:\Users\15-DK1031LA\Downloads\borrar\dataset_combinado"
    
    train_data, validation_data, test_data = cargar_datos(carpeta_origen)
    
    modelo = crear_modelo(num_clases=len(train_data.class_indices))
    
    historial = entrenar_modelo(modelo, train_data, validation_data, epochs=20)
    
    modelo.save('prueba.h5')
    print("[✅] Modelo guardado como 'modelo_signos_estaticos.h5'")
