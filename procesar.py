import tensorflow as tf
import os
import numpy as np
import cv2

# Categorías (deben coincidir con las carpetas)
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

# Cargar los datos
imagenes, etiquetas = cargar_datos('Data')

# Shuffle de los datos
indice = np.arange(len(imagenes))
np.random.shuffle(indice)
imagenes = imagenes[indice]
etiquetas = etiquetas[indice]

# Dividir entre entrenamiento y prueba
from sklearn.model_selection import train_test_split
imagenes_entrenamiento, imagenes_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(imagenes, etiquetas, test_size=0.2)

# Crear datasets de TensorFlow
datos_entrenamiento = tf.data.Dataset.from_tensor_slices((imagenes_entrenamiento, etiquetas_entrenamiento)).batch(32)
datos_pruebas = tf.data.Dataset.from_tensor_slices((imagenes_prueba, etiquetas_prueba)).batch(32)
print("Datos de entrenamiento:", imagenes_entrenamiento.shape)
print("Datos de prueba:", imagenes_prueba.shape)
print("¡Procesamiento completado!")
