import csv
from fastapi import FastAPI
from app.datos import cargar_residentes


app = FastAPI()
'''residentes = [
    {
        "id": 1,
        "full_name": "Juan Perez",
        "tower": 1,
        "apartment": 101
    },
    {
        "id": 2,
        "full_name": "Maria Gomez",
        "tower": 2,
        "apartment": 305
    }
]
'''
@app.get("/")
def root():
    return {"message": "Bienvenido a SmartBuilding API"}

@app.get("/residentes")
def get_residentes():
    return residentes

@app.get("/residentes/{residente_id}")
def get_residente_por_id(residente_id: int):
    for residente in residentes:
        if residente["id"] == residente_id:
            return residente
'''
residentes = []
with open("data/residentes_san_carlos.csv", encoding="utf-8-sig") as archivo:
    lector = csv.DictReader(archivo)
    for fila in lector:
        residentes.append({
            "id": int(fila["id"]),
            "full_name": fila["nombre"],
            "tower": int(fila["torre"]),
            "apartment": int(fila["apartamento"])
        })
'''
residentes = cargar_residentes()