import csv

def cargar_residentes():
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
    return residentes
