import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Ruta de los datos
carpeta_datos = "datos_gestos"
archivos = [f for f in os.listdir(carpeta_datos) if f.endswith(".csv")]

X = []
y = []

# Cargar datos de cada archivo
for archivo in archivos:
    etiqueta = archivo.replace(".csv", "")
    ruta = os.path.join(carpeta_datos, archivo)
    with open(ruta, "r") as f:
        lector = csv.reader(f)
        for fila in lector:
            if len(fila) == 63:  # 21 puntos * 3 coordenadas (x, y, z)
                X.append([float(i) for i in fila])
                y.append(etiqueta)

# Validación
if len(X) == 0:
    print("⚠️ No se encontraron datos válidos. ¿Capturaste los gestos?")
    exit()

# Convertir a arrays
X = np.array(X)
y = np.array(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo SVM
modelo = svm.SVC(kernel='linear')
modelo.fit(X_train, y_train)

# Evaluar modelo
y_pred = modelo.predict(X_test)
print("✅ Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\n✅ Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Guardar modelo
os.makedirs("modelos", exist_ok=True)
ruta_modelo = "modelos/modelo_gestos.pkl"
joblib.dump(modelo, ruta_modelo)
print(f"\n✅ Modelo guardado exitosamente en: {ruta_modelo}")
