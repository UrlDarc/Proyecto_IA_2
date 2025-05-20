import cv2
import mediapipe as mp
import csv
import os

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Crear carpeta para guardar datos si no existe
os.makedirs('datos_gestos', exist_ok=True)

# Nombre del gesto que se está capturando
gesto_nombre = input("Nombre del gesto: ")
archivo_csv = f'datos_gestos/{gesto_nombre}.csv'

# Abrir archivo CSV una vez, fuera del bucle
f = open(archivo_csv, mode='w', newline='')
writer = csv.writer(f)
contador = 0

# Abrir cámara
cap = cv2.VideoCapture(0)
print("Presiona 's' para guardar un gesto, 'q' para salir.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(img_rgb)

        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Captura de Gestos", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if resultados.multi_hand_landmarks:
                landmarks = resultados.multi_hand_landmarks[0]
                fila = []
                for lm in landmarks.landmark:
                    fila.extend([lm.x, lm.y, lm.z])
                if len(fila) == 63:
                    writer.writerow(fila)
                    contador += 1
                    print(f"✅ Gesto guardado ({contador})")
                else:
                    print("⚠️ Puntos incompletos, no se guardó.")
            else:
                print("⚠️ Mano no detectada.")

        elif key == ord('q'):
            print("Saliendo...")
            break
finally:
    f.close()
    cap.release()
    cv2.destroyAllWindows()
