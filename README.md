# SmartBuilding API — Suite Completa de Visión por Computador (Retos 1 al 7) 🏢🎥

Backend para la administración de edificios residenciales (**Conjunto Residencial San Carlos**) desarrollado con **Python**, **FastAPI** y **OpenCV**, incorporando los **7 retos de visión por computador** conectados con el directorio de residentes (`data/residentes_san_carlos.csv`).

---

## 🌿 Estructura de Ramas

- **`main`**: Implementación base de residentes + **Reto 1 (Demografía y vestimenta)** únicamente.
- **`feature/retos-vision-completos`** *(esta rama)*: Implementación integral de los **7 retos de visión por computador** + Dashboard interactivo multi-cámara + videos de prueba incluidos en `data/videos/` + suite completa de pruebas `pytest` (21 tests).

---

## 🧠 Los 7 Retos Implementados

| # | Reto | Módulo | Video de prueba incluido | Endpoint API |
|---|---|---|---|---|
| **1** | **Demografía y vestimenta** (sexo aparente, rango de edad, tipo y color de prenda superior e inferior) | `app/vision/demografia_vestimenta.py` | `data/videos/reto1_demografia_vestimenta.mp4` | `POST /vision/reto1/demografia-vestimenta` |
| **2** | **Conteo de entradas y salidas** (múltiples líneas, cada sentido y detección de hora pico) | `app/vision/conteo_entradas_salidas.py` | `data/videos/reto2_conteo_flujo.mp4` | `POST /vision/reto2/conteo-entradas-salidas` |
| **3** | **Merodeo** (alerta deambulación anómala sin alertar a quien espera, conversa o pasa) | `app/vision/merodeo.py` | `data/videos/reto3_merodeo.mp4` | `POST /vision/reto3/merodeo` |
| **4** | **Objeto abandonado** (alerta cuando dejan bolso, maleta o paquete y se alejan — dataset ABODA) | `app/vision/objeto_abandonado.py` | `data/videos/reto4_objeto_abandonado.mp4` | `POST /vision/reto4/objeto-abandonado` |
| **5** | **Lectura de placas** (carros y motos, día y noche con CLAHE, 1 lectura única por vehículo + cruce con residentes) | `app/vision/lectura_placas.py` | `data/videos/reto5_lectura_placas.mp4` | `POST /vision/reto5/lectura-placas` |
| **6** | **Caídas y comportamientos de riesgo** (detecta caída sin levantarse, sin confundir con sentarse o agacharse) | `app/vision/caidas_riesgo.py` | `data/videos/reto6_caidas_riesgo.mp4` | `POST /vision/reto6/caidas-riesgo` |
| **7** | **Correspondencia en casillero** (detecta paquetes o recibos en casillero y reporta número de apto y residentes) | `app/vision/correspondencia_casillero.py` | `data/videos/reto7_casilleros.mp4` | `POST /vision/reto7/correspondencia-casillero` |

---

## 📚 Bases de Datos y Repositorios de Videos de Vigilancia

1. **Intel IoT DevKit Sample Videos** (`https://github.com/intel-iot-devkit/sample-videos`):
   - `people-detection.mp4`, `one-by-one-person-detection.mp4`, `face-demographics-walking.mp4`, `car-detection.mp4`.
2. **ABODA (Abandoned Object Dataset)** (`https://github.com/kevinlin311tw/ABODA`):
   - `video1.avi` a `video11.avi` para escenarios reales de objetos abandonados e interacción en pasillos.
3. **PETA / PA-100K / RAP** (`https://github.com/xh-liu/HydraPlus-Net`):
   - Datasets de referencia para *Pedestrian Attribute Recognition* (edad, sexo aparente, vestimenta).
4. **UR Fall Detection Dataset (URFD)** (`http://fenix.ur.edu.pl/~mkepski/ds/uf.html`):
   - Secuencias de caídas vs actividades cotidianas (sentarse, agacharse).

Puedes descargar los videos originales completos ejecutando:
```bash
python scripts/descargar_videos_prueba.py
```

---

## 🚀 Ejecución y Pruebas

```bash
# Instalar dependencias
pip install -r requirements.txt

# Correr las 21 pruebas automatizadas (TDD)
pytest -v

# Levantar el servidor FastAPI
uvicorn app.main:app --reload
```

- **Dashboard Multi-Cámara (Retos 1 al 7)**: `http://127.0.0.1:8000/dashboard`
- **Swagger UI**: `http://127.0.0.1:8000/docs`
