"""Carga inicial de SmartBuilding desde el CSV a las ocho tablas existentes.

Ubicar este archivo junto al .env, en la raíz de smartbuilding-api.
Dependencias: psycopg[binary] y python-dotenv.
Uso: python cargar_csv.py
Validación sin conexión: python cargar_csv.py --validar

La carga exige que las ocho tablas estén vacías. No borra datos ni modifica
el esquema. Todas las inserciones se confirman juntas o se revierten.
El id del CSV se valida, pero PostgreSQL genera sus propios identificadores.
"""

import argparse
import csv
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
RUTA_CSV = BASE_DIR / "data" / "residentes_san_carlos.csv"
NOMBRE_EDIFICIO = "San Carlos"
TABLAS = (
    "edificio", "torre", "apartamento", "persona", "mascota",
    "casillero", "parqueadero", "vehiculo",
)
COLUMNAS = (
    "id", "nombre", "tipo", "torre", "apartamento", "telefono", "email",
    "mascota", "casillero", "parqueadero_1", "placa_1",
    "parqueadero_2", "placa_2",
)
CAMPOS_COMPARTIDOS = (
    "mascota", "casillero", "parqueadero_1", "placa_1",
    "parqueadero_2", "placa_2",
)


def leer_csv(ruta):
    """Lee todas las columnas y limpia espacios exteriores."""
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if not lector.fieldnames or len(set(lector.fieldnames)) != len(lector.fieldnames):
            raise ValueError("El CSV no tiene encabezados válidos y únicos.")
        faltantes = set(COLUMNAS) - set(lector.fieldnames)
        if faltantes:
            raise ValueError(f"Faltan columnas: {', '.join(sorted(faltantes))}.")
        filas = []
        for numero, fila in enumerate(lector, start=2):
            if None in fila or any(valor is None for valor in fila.values()):
                raise ValueError(f"Fila {numero}: cantidad incorrecta de campos.")
            filas.append({campo: fila[campo].strip() for campo in COLUMNAS})
    if not filas:
        raise ValueError("El CSV no contiene registros.")
    return filas


def preparar_datos(filas):
    """Agrupa datos compartidos por torre/apartamento y detecta conflictos."""
    apartamentos = {}
    ids_csv = set()
    casilleros = set()
    parqueaderos = set()
    placas = set()
    for numero, fila in enumerate(filas, start=2):
        for campo in COLUMNAS:
            if campo not in ("telefono", "email") and not fila[campo]:
                raise ValueError(f"Fila {numero}: el campo {campo} está vacío.")
        try:
            id_csv = int(fila["id"])
            torre = int(fila["torre"])
            numero_apartamento = int(fila["apartamento"])
        except ValueError as error:
            raise ValueError(f"Fila {numero}: id, torre y apartamento deben ser enteros.") from error
        if id_csv <= 0 or torre <= 0 or numero_apartamento // 100 <= 0:
            raise ValueError(f"Fila {numero}: id, torre o numeración del apartamento inválidos.")
        if id_csv in ids_csv:
            raise ValueError(f"Fila {numero}: id del CSV repetido ({id_csv}).")
        ids_csv.add(id_csv)
        if fila["tipo"] not in ("propietario", "residente"):
            raise ValueError(f"Fila {numero}: tipo de persona inválido.")

        # En este conjunto de datos, 101 pertenece al piso 1, 801 al piso 8.
        clave = (torre, fila["apartamento"])
        if clave in apartamentos:
            anterior = apartamentos[clave]
            if any(fila[c] != anterior[c] for c in CAMPOS_COMPARTIDOS):
                raise ValueError(f"Fila {numero}: datos compartidos inconsistentes para {clave}.")
            continue

        if fila["casillero"] in casilleros:
            raise ValueError(f"Fila {numero}: casillero asignado a varios apartamentos.")
        casilleros.add(fila["casillero"])
        for indice in (1, 2):
            codigo = fila[f"parqueadero_{indice}"]
            placa = fila[f"placa_{indice}"]
            if codigo in parqueaderos:
                raise ValueError(f"Fila {numero}: parqueadero repetido ({codigo}).")
            parqueaderos.add(codigo)
            if placa.casefold() != "libre":
                if placa.upper() in placas:
                    raise ValueError(f"Fila {numero}: placa asignada a varios parqueaderos.")
                placas.add(placa.upper())
        apartamentos[clave] = fila

    esperados = {
        "edificio": 1,
        "torre": len({clave[0] for clave in apartamentos}),
        "apartamento": len(apartamentos),
        "persona": len(filas),
        "mascota": sum(f["mascota"].casefold() != "no posee" for f in apartamentos.values()),
        "casillero": len(casilleros),
        "parqueadero": len(parqueaderos),
        "vehiculo": len(placas),
    }
    return apartamentos, esperados


def conectar():
    """Reutiliza las credenciales del .env junto a este script."""
    load_dotenv(BASE_DIR / ".env")
    for variable in ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"):
        if not os.environ.get(variable):
            raise ValueError(f"Falta configurar {variable} en el entorno o en .env.")
    return psycopg.connect(
        host="localhost",
        port=5432,
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=5,
    )


def insertar_id(cursor, consulta, valores):
    """Devuelve el identificador generado por PostgreSQL con RETURNING."""
    cursor.execute(consulta, valores)
    return cursor.fetchone()[0]


def cargar_datos(filas, apartamentos, esperados):
    # El contexto confirma al terminar y revierte si se produce una excepción.
    with conectar() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute("SET LOCAL lock_timeout = '5s'")
            cursor.execute("SET LOCAL search_path = public")
            # Impide que otra carga escriba entre la comprobación y las inserciones.
            cursor.execute(
                "LOCK TABLE edificio, torre, apartamento, persona, mascota, "
                "casillero, parqueadero, vehiculo IN SHARE ROW EXCLUSIVE MODE"
            )
            for tabla in TABLAS:
                # tabla procede exclusivamente de la constante TABLAS.
                cursor.execute(f"SELECT COUNT(*) FROM {tabla}")
                if cursor.fetchone()[0] != 0:
                    raise ValueError(
                        f"La tabla {tabla} contiene datos. La carga inicial requiere "
                        "las ocho tablas vacías; no se modificó ningún registro."
                    )

            id_edificio = insertar_id(
                cursor,
                "INSERT INTO edificio (nombre) VALUES (%s) RETURNING id_edificio",
                (NOMBRE_EDIFICIO,),
            )
            torres = {}
            ids_apartamentos = {}
            for (numero_torre, numero_apartamento), fila in apartamentos.items():
                if numero_torre not in torres:
                    torres[numero_torre] = insertar_id(
                        cursor,
                        "INSERT INTO torre (id_edificio, numero) VALUES (%s, %s) RETURNING id_torre",
                        (id_edificio, numero_torre),
                    )
                id_apartamento = insertar_id(
                    cursor,
                    "INSERT INTO apartamento (id_torre, numero, piso) "
                    "VALUES (%s, %s, %s) RETURNING id_apartamento",
                    (torres[numero_torre], numero_apartamento, int(numero_apartamento) // 100),
                )
                ids_apartamentos[(numero_torre, numero_apartamento)] = id_apartamento
                cursor.execute(
                    "INSERT INTO casillero (id_apartamento, codigo) VALUES (%s, %s)",
                    (id_apartamento, fila["casillero"]),
                )
                if fila["mascota"].casefold() != "no posee":
                    cursor.execute(
                        "INSERT INTO mascota (id_apartamento, tipo) VALUES (%s, %s)",
                        (id_apartamento, fila["mascota"]),
                    )
                for indice in (1, 2):
                    id_parqueadero = insertar_id(
                        cursor,
                        "INSERT INTO parqueadero (id_apartamento, codigo) "
                        "VALUES (%s, %s) RETURNING id_parqueadero",
                        (id_apartamento, fila[f"parqueadero_{indice}"]),
                    )
                    placa = fila[f"placa_{indice}"]
                    if placa.casefold() != "libre":
                        cursor.execute(
                            "INSERT INTO vehiculo (id_parqueadero, placa) VALUES (%s, %s)",
                            (id_parqueadero, placa.upper()),
                        )

            for fila in filas:
                clave = (int(fila["torre"]), fila["apartamento"])
                cursor.execute(
                    "INSERT INTO persona (id_apartamento, nombre, telefono, email, tipo) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (ids_apartamentos[clave], fila["nombre"], fila["telefono"] or None,
                     fila["email"] or None, fila["tipo"]),
                )

            # Comprobación dentro de la transacción: un desacuerdo también revierte.
            for tabla, esperado in esperados.items():
                cursor.execute(f"SELECT COUNT(*) FROM {tabla}")
                actual = cursor.fetchone()[0]
                if actual != esperado:
                    raise ValueError(f"Cantidad incorrecta en {tabla}: {actual}; se esperaba {esperado}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validar", action="store_true", help="Validar el CSV sin conectarse ni insertar.")
    argumentos = parser.parse_args()
    filas = leer_csv(RUTA_CSV)
    apartamentos, esperados = preparar_datos(filas)
    print(f"CSV validado: {len(filas)} filas.")
    if not argumentos.validar:
        cargar_datos(filas, apartamentos, esperados)
        print("Carga completada y confirmada en PostgreSQL.")
    else:
        print("Solo validación: no se abrió conexión ni se insertaron datos.")
    print("Cantidades esperadas:" if argumentos.validar else "Registros verificados:")
    for tabla, cantidad in esperados.items():
        print(f"  {tabla}: {cantidad}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error, psycopg.Error) as error:
        print(f"ERROR: {error}")
        print("La carga no se completó. No se confirmaron inserciones de esta ejecución.")
        raise SystemExit(1)
