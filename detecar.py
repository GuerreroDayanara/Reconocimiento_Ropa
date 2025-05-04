import tensorflow as tf
import numpy as np
import cv2

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model('modelo_exportado.h5')

# Categorías
categorias = ['Camiseta', 'Pantalon', 'Desconocido']

# Función para realizar la predicción con un umbral de certeza
def predecir(imagen):
    # Realizar la predicción
    prediccion = modelo.predict(imagen)
    probabilidad_maxima = np.max(prediccion)  # Obtener la probabilidad más alta
    clase_predicha = np.argmax(prediccion)  # Obtener la clase con mayor probabilidad
    
    print(f'Probabilidades: {prediccion}')  # Para ver las probabilidades de todas las clases
    print(f'Probabilidad máxima: {probabilidad_maxima}, Clase predicha: {clase_predicha}')
    
    # Si la probabilidad es menor que un umbral, consideramos "Desconocido"
    umbral = 0.5  # Umbral de certeza ajustado a 50%
    if probabilidad_maxima < umbral:
        return "Desconocido"
    else:
        return categorias[clase_predicha]

# Abrir cámara
cap = cv2.VideoCapture(0)  # 0 para la webcam principal

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar imagen de la cámara")
        break

    # Preprocesar la imagen
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    img = cv2.resize(img, (28, 28))  # Redimensionar
    img = img / 255.0  # Normalizar
    img = img.reshape(1, 28, 28, 1)  # Ajustar forma para el modelo

    nombre_clase = predecir(img)
    cv2.putText(frame, f'Ropa Visible: {nombre_clase}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Reconocimiento de ropa', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
