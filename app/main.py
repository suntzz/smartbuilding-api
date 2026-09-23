from fastapi import FastAPI, HTTPException
from app.datos import cargar_residentes

app = FastAPI()

residentes = cargar_residentes()

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
    raise HTTPException(status_code=404, detail="Residente no encontrado")