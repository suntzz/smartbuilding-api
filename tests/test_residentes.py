"""Pruebas de integración contra PostgreSQL 
Requieren Docker encendido, .env en la raíz y la carga de 128 personas.
    python -m pytest tests/test_residentes.py -v
"""

import os
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.main import app


BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def base_real():
    """Conexión independiente para contrastar las respuestas de la API."""
    load_dotenv(BASE_DIR / ".env")
    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=5,
    ) as conexion:
        conexion.execute("SET TRANSACTION READ ONLY")
        yield conexion


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as cliente:
        yield cliente


def consultar_personas(conexion):
    """Lee los datos sin reutilizar las funciones de consulta de la API."""
    with conexion.cursor() as cursor:
        cursor.execute("""
            SELECT p.id_persona, p.nombre, t.numero, a.numero
            FROM public.persona AS p
            JOIN public.apartamento AS a
                ON a.id_apartamento = p.id_apartamento
            JOIN public.torre AS t
                ON t.id_torre = a.id_torre
            ORDER BY p.id_persona
        """)
        return [
            {"id": id_persona, "full_name": nombre,
             "tower": torre, "apartment": int(apartamento)}
            for id_persona, nombre, torre, apartamento in cursor.fetchall()
        ]


def test_get_residentes_postgresql(client, base_real):
    esperados = consultar_personas(base_real)
    assert len(esperados) == 128, "Se requiere la carga inicial de 128 personas."
    respuesta = client.get("/residentes")
    assert respuesta.status_code == 200
    assert respuesta.json() == esperados


def test_get_residente_10_postgresql(client, base_real):
    esperado = next(
        (persona for persona in consultar_personas(base_real) if persona["id"] == 10),
        None,
    )
    assert esperado is not None, "No existe id_persona=10 en la base cargada."
    respuesta = client.get("/residentes/10")
    assert respuesta.status_code == 200
    assert respuesta.json() == esperado


def test_get_residente_inexistente(client, base_real):
    with base_real.cursor() as cursor:
        cursor.execute("SELECT COALESCE(MAX(id_persona), 0) + 1 FROM public.persona")
        id_inexistente = cursor.fetchone()[0]
    respuesta = client.get(f"/residentes/{id_inexistente}")
    # Conserva el comportamiento previo; responder 404 sería otro cambio.
    assert respuesta.status_code == 200
    assert respuesta.json() is None


def test_get_residente_id_invalido(client):
    respuesta = client.get("/residentes/abc")
    assert respuesta.status_code == 422
