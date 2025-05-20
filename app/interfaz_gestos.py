import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import joblib

ventana = tk.Tk()
ventana.title("Simulador Multimedia con Gestos")
ventana.geometry("800x600")
ventana.configure(bg="#1e1e1e")

# Cargar imágenes de canciones
canciones = [
    {
        "nombre": "Canción 1",
        "imagen": ImageTk.PhotoImage(Image.open("canciones/cancion1.png").resize((300, 400)), master=ventana)
    },
    {
        "nombre": "Canción 2",
        "imagen": ImageTk.PhotoImage(Image.open("canciones/cancion2.png").resize((300, 400)), master=ventana)
    }
]

indice_cancion = 0

# Label para portada y título
label_portada = tk.Label(ventana, image=canciones[indice_cancion]["imagen"], bg="#1e1e1e")
label_portada.pack(side="left", padx=20, pady=20)

label_titulo = tk.Label(ventana, text=canciones[indice_cancion]["nombre"], font=("Helvetica", 20), bg="#1e1e1e", fg="#00cccc")
label_titulo.pack(side="left", pady=20)

# Cargar iconos para gestos
iconos_paths = {
    "play": "iconos/play.png",
    "pause": "iconos/pause.png",
    "next": "iconos/next.png",
    "volume_up": "iconos/volume_up.png",
    "volume_down": "iconos/volume_down.png",
    "default": "iconos/play.png"
}

iconos_imgs = {}
for key, path in iconos_paths.items():
    img = Image.open(path).resize((150, 150))
    iconos_imgs[key] = ImageTk.PhotoImage(img, master=ventana)

label_icono = tk.Label(ventana, image=iconos_imgs["default"], bg="#1e1e1e")
label_icono.pack(side="right", padx=40, pady=20)

modelo = joblib.load("../modelos/modelo_gestos.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

acciones_map = {
    "palma": "pause",
    "punio": "next",
    "senal": "volume_up",
    "dedos": "volume_down"
}

def actualizar_cancion(siguiente=True):
    global indice_cancion
    if siguiente:
        indice_cancion = (indice_cancion + 1) % len(canciones)
    else:
        indice_cancion = (indice_cancion - 1) % len(canciones)
    label_portada.config(image=canciones[indice_cancion]["imagen"])
    label_titulo.config(text=canciones[indice_cancion]["nombre"])

cap = cv2.VideoCapture(0)

def procesar_frame():
    ret, frame = cap.read()
    if not ret:
        ventana.after(30, procesar_frame)
        return

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(img_rgb)

    icono_actual = "default"
    cambio_cancion = False

    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fila = []
            for lm in hand_landmarks.landmark:
                fila.extend([lm.x, lm.y, lm.z])

            if len(fila) == 63:
                entrada = np.array(fila).reshape(1, -1)
                prediccion = modelo.predict(entrada)[0]
                icono_actual = acciones_map.get(prediccion, "default")

                if icono_actual == "next":
                    actualizar_cancion(siguiente=True)
                    cambio_cancion = True

    label_icono.config(image=iconos_imgs[icono_actual])

    cv2.imshow("Camara y Detección", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        ventana.destroy()
        return

    # Se pausa un tiempo para cambio de iconos
    delay = 1000 if cambio_cancion else 10
    ventana.after(delay, procesar_frame)

ventana.after(10, procesar_frame)
ventana.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), cv2.destroyAllWindows(), ventana.destroy()))
ventana.mainloop()
