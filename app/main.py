from fastapi import FastAPI

from app.datos import cargar_residentes, obtener_residente_por_id


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Bienvenido a SmartBuilding API"}


@app.get("/residentes")
def get_residentes():
    return cargar_residentes()


@app.get("/residentes/{residente_id}")
def get_residente_por_id(residente_id: int):
    # Null si el ID no existe.
    return obtener_residente_por_id(residente_id)


# Visión por computador (opcional): solo si están instaladas sus librerías
try:
    from app.api_vision import registrar_vision
except ModuleNotFoundError:
    pass
else:
    registrar_vision(app)
