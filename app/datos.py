import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_RESIDENTES_PATH = BASE_DIR / "data" / "residentes_san_carlos.csv"


def cargar_residentes(ruta_csv: str | Path | None = None) -> list[dict]:
    ruta = Path(ruta_csv) if ruta_csv else CSV_RESIDENTES_PATH
    residentes = []
    with open(ruta, encoding="utf-8-sig") as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            residente = {
                "id": int(fila["id"]),
                "full_name": fila["nombre"],
                "tower": int(fila["torre"]),
                "apartment": int(fila["apartamento"]),
                "tipo": fila["tipo"],
                "telefono": fila["telefono"],
                "email": fila["email"],
            }
            if "mascota" in fila:
                residente["mascota"] = fila["mascota"]
            if "casillero" in fila:
                residente["casillero"] = fila["casillero"]
            if "parqueadero_1" in fila:
                residente["parqueadero_1"] = fila["parqueadero_1"]
            if "placa_1" in fila:
                residente["placa_1"] = fila["placa_1"]
            if "parqueadero_2" in fila:
                residente["parqueadero_2"] = fila["parqueadero_2"]
            if "placa_2" in fila:
                residente["placa_2"] = fila["placa_2"]
            residentes.append(residente)
    return residentes
