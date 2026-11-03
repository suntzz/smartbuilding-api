# SmartBuilding API — Reto 1: Demografía y Vestimenta 🏢🎥

🌐 **Demo en vivo:** https://suntzz.github.io/smartbuilding-api/

Backend para la administración de edificios residenciales (**Conjunto Residencial San Carlos**) desarrollado con **Python**, **FastAPI** y **OpenCV**, incorporando el **Reto 1 de Visión por Computador: Demografía y vestimenta**.

> **Nota de ramas:**
> - **`main`**: Implementación base de residentes + **Reto 1 (Demografía y vestimenta)** con videos reales, dashboard interactivo y suite de pruebas `pytest`.
> - **`feature/retos-vision-completos`**: Implementación completa de los **7 retos de visión por computador** (Demografía y vestimenta, Conteo de entradas y salidas, Merodeo, Objeto abandonado, Lectura de placas, Caídas y comportamientos de riesgo, y Correspondencia en casillero).

---

## 🎯 Reto 1: Demografía y Vestimenta

Para cada persona que pasa frente a la cámara de seguridad del edificio, el módulo [`app/vision/demografia_vestimenta.py`](app/vision/demografia_vestimenta.py) detecta, rastrea (`person_id`) y estima:

1. **Sexo aparente**: `Masculino` | `Femenino`
2. **Rango de edad**: `Infantil (0-14)` | `Joven (15-29)` | `Adulto (30-54)` | `Adulto Mayor (55+)`
3. **Prenda superior**:
   - **Tipo**: `camiseta`, `camisa`, `chaqueta`, `buso`, `blusa`
   - **Color dominante**: `negro`, `blanco`, `gris`, `rojo`, `azul`, `azul oscuro`, `verde`, `amarillo`, `naranja`, `morado`, `rosado`, `cafe`, `beige` + código hexadecimal (`#RRGGBB`)
4. **Prenda inferior**:
   - **Tipo**: `pantalon`, `jean`, `pantaloneta`, `falda`
   - **Color dominante**: nombre en español + código hexadecimal (`#RRGGBB`)

---

## 📹 Videos de Prueba y Bases de Datos de Referencia

El repositorio incluye clips reales de cámaras de vigilancia en `data/videos/`:
- `data/videos/reto1_demografia_vestimenta.mp4`: Cámara cenital/lobby con múltiples peatones de cuerpo completo (derivado de [`intel-iot-devkit/sample-videos`](https://github.com/intel-iot-devkit/sample-videos) — `people-detection.mp4`).
- `data/videos/reto1_corredor_frontal.mp4`: Cámara de pasillo frontal (`face-demographics-walking.mp4`).

---

## 🚀 Instalación y Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar pruebas unitarias y de integración (TDD)
pytest -v

# 3. Iniciar el servidor FastAPI
uvicorn app.main:app --reload
```

- **Dashboard Interactivo con Video en Vivo**: `http://127.0.0.1:8000/dashboard`
- **Documentación Swagger OpenAPI**: `http://127.0.0.1:8000/docs`
