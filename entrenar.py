import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Categorías
categorias = ['Camiseta', 'Pantalon', 'Zapatos' ,'Desconocido']

num_clases = len(categorias)

def cargar_datos(directorio):
    imagenes = []
    etiquetas = []
    for idx, categoria in enumerate(categorias):
        ruta_categoria = os.path.join(directorio, categoria)
        for archivo in os.listdir(ruta_categoria):
            img_path = os.path.join(ruta_categoria, archivo)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = img / 255.0
            imagenes.append(img.reshape(28,28,1))
            etiquetas.append(idx)
    return np.array(imagenes), np.array(etiquetas)

# Cargar datos
imagenes, etiquetas = cargar_datos('Data')

# Shuffle
indices = np.arange(len(imagenes))
np.random.shuffle(indices)
imagenes = imagenes[indices]
etiquetas = etiquetas[indices]

# Dividir datos
imagenes_entrenamiento, imagenes_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(imagenes, etiquetas, test_size=0.2)

# Crear datasets
datos_entrenamiento = tf.data.Dataset.from_tensor_slices((imagenes_entrenamiento, etiquetas_entrenamiento)).batch(32)
datos_pruebas = tf.data.Dataset.from_tensor_slices((imagenes_prueba, etiquetas_prueba)).batch(32)

# Crear modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_clases, activation='softmax')
])

# Compilar modelo
modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Entrenar
modelo.fit(datos_entrenamiento, epochs=10, validation_data=datos_pruebas)

# Guardar modelo
modelo.save('modelo_exportado.h5')
print("Modelo entrenado y exportado exitosamente!")
