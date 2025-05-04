import cv2
import os
# Configura las categorías que vas a capturar
categorias = ['Camiseta', 'Pantalon', 'Zapatos' ,'Desconocido']

# Crea carpetas si no existen
for categoria in categorias:
    os.makedirs(f'Data/{categoria}', exist_ok=True)

camara = cv2.VideoCapture(0)

print("\nPresiona 'c' para capturar una imagen.")
print("Presiona 'ESC' para salir.\n")

categoria_actual = input(f"Selecciona una categoría {categorias}: ")

contador = 0

while True:
    ret, frame = camara.read()
    if not ret:
        print("Error al acceder a la cámara")
        break
    cv2.imshow(f"Capturando: {categoria_actual}", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC para salir
        print("Saliendo...")
        break
    elif k%256 == ord('c'):
        img_name = f"Data/{categoria_actual}/{categoria_actual}_{contador}.png"
        cv2.imwrite(img_name, frame)
        print(f"Imagen guardada: {img_name}")
        contador += 1

camara.release()
cv2.destroyAllWindows()
