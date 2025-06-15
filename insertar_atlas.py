# Conexión a MongoDB Atlas
from pymongo import MongoClient
import pandas as pd

# 1. Cargar datos
datos = pd.read_csv("prueba_atlas_enfriadores.csv")

# 2. Conectar a Atlas
cliente = MongoClient("mongodb+srv://fridgyUser:fridgyPassword@cluster0.tltyk0b.mongodb.net/")
bd = cliente["monitoreo_enfriadores"]
coleccion = bd["metricas_tiempo_real"]

# 3. Insertar datos
coleccion.insert_many(datos.to_dict('records'))
print("✅ Datos insertados en Atlas. CSV generado: 'prueba_atlas_enfriadores.csv'")