# Prueba de Concepto para MongoDB Atlas

## Objetivo
Demostrar la integración de MongoDB Atlas con dispositivos IoT M5GO para monitoreo predictivo de enfriadores.

## Archivos Incluidos
1. `prueba_atlas_enfriadores.csv` - Datos de ejemplo con:
   - Geolocalización
   - Mediciones de sensores
   - Estado de mantenimiento

2. `insertar_atlas.py` - Script para insertar datos en Atlas.

## Cómo Verificar
1. Ejecutar el script Python (requiere `pymongo` y `pandas`)
2. Consultar en Atlas:
   ```javascript
   db.metricas_tiempo_real.find({
     necesita_mantenimiento: "urgente",
     ubicacion: {
       $nearSphere: {
         $geometry: { type: "Point", coordinates: [-74.006, 40.7128] }
       }
     }
   })