import cv2
import mediapipe as mp
import numpy as np
import joblib


# Cargar el modelo entrenado
modelo = joblib.load("modelos/modelo_gestos.pkl")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Diccionario de acciones asociadas
acciones = {
    "palma": "⏯️ Reproducir / Pausar",
    "punio": "⏭️ Siguiente",
    "senal": "🔊 Subir volumen",
    "dedos": "🔉 Bajar volumen"
}

# Iniciar webcam
cap = cv2.VideoCapture(0)
print("Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(img_rgb)

    prediccion = "Sin gesto"
    accion = ""

    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fila = []
            for lm in hand_landmarks.landmark:
                fila.extend([lm.x, lm.y, lm.z])

            if len(fila) == 63:
                entrada = np.array(fila).reshape(1, -1)
                prediccion = modelo.predict(entrada)[0]
                accion = acciones.get(prediccion, "Sin acción")

    # Mostrar en pantalla
    cv2.putText(frame, f"Gesto: {prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if accion:
        cv2.putText(frame, f"Accion: {accion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    cv2.imshow("Reconocimiento de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
