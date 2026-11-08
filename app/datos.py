"""Consultas de residentes a PostgreSQL"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row


BASE_DIR = Path(__file__).resolve().parent.parent

# Los alias conservan las claves JSON de la API existente.
CONSULTA_RESIDENTES = """
    SELECT
        p.id_persona AS id,
        p.nombre AS full_name,
        t.numero AS tower,
        CAST(a.numero AS INTEGER) AS apartment
    FROM public.persona AS p
    JOIN public.apartamento AS a
        ON p.id_apartamento = a.id_apartamento
    JOIN public.torre AS t
        ON a.id_torre = t.id_torre
"""


def conectar():
    """Abre una conexión con las mismas credenciales usadas en la carga."""
    load_dotenv(BASE_DIR / ".env")
    return psycopg.connect(
        host="localhost",
        port=5432,
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=5,
        row_factory=dict_row,
    )


def cargar_residentes():
    """Consulta la lista en cada llamada y cierra la conexión al terminar."""
    with conectar() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(CONSULTA_RESIDENTES + " ORDER BY p.id_persona")
            return cursor.fetchall()


def obtener_residente_por_id(residente_id):
    """Busca por el identificador generado por PostgreSQL, usando un parámetro."""
    with conectar() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                CONSULTA_RESIDENTES + " WHERE p.id_persona = %s",
                (residente_id,),
            )
            return cursor.fetchone()
