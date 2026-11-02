import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.vision.caidas_riesgo import DetectorCaidasRiesgo
from app.vision.conteo_entradas_salidas import ContadorEntradasSalidas
from app.vision.correspondencia_casillero import DetectorCorrespondenciaCasillero
from app.vision.lectura_placas import LectorPlacasVehiculares, mejorar_iluminacion_nocturna
from app.vision.merodeo import DetectorMerodeo
from app.vision.objeto_abandonado import DetectorObjetoAbandonado

client = TestClient(app)


# =====================================================================
# Pruebas Reto 2: Conteo de entradas y salidas
# =====================================================================
def test_reto2_conteo_entradas_salidas_y_hora_pico():
    contador = ContadorEntradasSalidas()
    res = contador.analizar_video("data/videos/reto2_conteo_flujo.mp4", cada_n_frames=2)

    assert res["reto"] == "2. Conteo de entradas y salidas"
    assert res["total_entradas"] >= 2
    assert res["total_salidas"] >= 2
    assert res["hubo_hora_pico"] is True
    assert res["max_personas_simultaneas"] >= 3
    assert "linea_acceso_principal" in res["conteo_por_linea"]


def test_reto2_endpoint_api():
    resp = client.post("/vision/reto2/conteo-entradas-salidas")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_entradas"] >= 2
    assert data["total_salidas"] >= 2


# =====================================================================
# Pruebas Reto 3: Merodeo
# =====================================================================
def test_reto3_clasificar_comportamiento_trayectoria():
    detector = DetectorMerodeo(umbral_frames_merodeo=10)

    # 1. Persona que simplemente pasa en línea recta -> NO alerta
    puntos_paso = [(50 + i * 15, 200) for i in range(15)]
    eval_paso = detector.clasificar_comportamiento_trayectoria(puntos_paso)
    assert eval_paso["comportamiento"] == "paso_normal"
    assert eval_paso["alerta"] is False

    # 2. Persona que espera quieta en un lugar -> NO alerta
    puntos_espera = [(200 + (i % 2), 200) for i in range(18)]
    eval_espera = detector.clasificar_comportamiento_trayectoria(puntos_espera)
    assert eval_espera["comportamiento"] == "esperando"
    assert eval_espera["alerta"] is False

    # 3. Personas que conversan en proximidad social -> NO alerta
    eval_conversa = detector.clasificar_comportamiento_trayectoria(
        puntos_espera, en_proximidad_social=True
    )
    assert eval_conversa["comportamiento"] == "conversando"
    assert eval_conversa["alerta"] is False

    # 4. Persona que deambula de ida y vuelta de forma anómala -> ALERTA MERODEO
    puntos_merodeo = (
        [(100 + i * 18, 220) for i in range(8)]
        + [(226 - i * 18, 225) for i in range(8)]
        + [(100 + i * 18, 230) for i in range(8)]
    )
    eval_merodeo = detector.clasificar_comportamiento_trayectoria(puntos_merodeo)
    assert eval_merodeo["comportamiento"] == "merodeo"
    assert eval_merodeo["alerta"] is True


def test_reto3_video_y_endpoint():
    resp = client.post("/vision/reto3/merodeo")
    assert resp.status_code == 200
    data = resp.json()
    assert data["reto"] == "3. Merodeo"
    assert data["total_alertas_merodeo"] >= 1
    assert data["conteo_estados_observados"]["paso_normal"] >= 1


# =====================================================================
# Pruebas Reto 4: Objeto abandonado
# =====================================================================
def test_reto4_objeto_abandonado_video_y_endpoint():
    resp = client.post("/vision/reto4/objeto-abandonado")
    assert resp.status_code == 200
    data = resp.json()
    assert data["reto"] == "4. Objeto abandonado"
    assert data["total_alertas_abandono"] >= 1
    tipos = {a["tipo_objeto"] for a in data["alertas"]}
    assert tipos.issubset({"bolso", "maleta", "paquete"})


# =====================================================================
# Pruebas Reto 5: Lectura de placas (Día y Noche, Carros y Motos, 1 lectura/vehículo)
# =====================================================================
def test_reto5_mejorar_iluminacion_dia_y_noche():
    img_noche = np.full((80, 120, 3), 25, dtype=np.uint8)
    _, cond_noche = mejorar_iluminacion_nocturna(img_noche)
    assert cond_noche == "noche"

    img_dia = np.full((80, 120, 3), 180, dtype=np.uint8)
    _, cond_dia = mejorar_iluminacion_nocturna(img_dia)
    assert cond_dia == "dia"


def test_reto5_lectura_placas_unica_por_vehiculo_y_cruce_residentes():
    resp = client.post("/vision/reto5/lectura-placas")
    assert resp.status_code == 200
    data = resp.json()

    assert data["reto"] == "5. Lectura de placas"
    assert data["total_vehiculos_leidos"] == 3
    placas = [v["placa"] for v in data["lecturas_unicas"]]
    # Sin lecturas duplicadas por vehículo
    assert len(placas) == len(set(placas))
    assert "KXW694" in placas
    assert "MTK48D" in placas
    assert "WGB442" in placas

    mapa = {v["placa"]: v for v in data["lecturas_unicas"]}
    assert mapa["KXW694"]["tipo_vehiculo"] == "carro"
    assert mapa["KXW694"]["condicion_luz"] == "dia"
    assert mapa["KXW694"]["autorizado"] is True
    assert mapa["KXW694"]["apartamento"] == 101

    assert mapa["MTK48D"]["tipo_vehiculo"] == "moto"
    assert mapa["MTK48D"]["autorizado"] is False

    assert mapa["WGB442"]["tipo_vehiculo"] == "carro"
    assert mapa["WGB442"]["condicion_luz"] == "noche"
    assert mapa["WGB442"]["autorizado"] is True
    assert mapa["WGB442"]["apartamento"] == 102


# =====================================================================
# Pruebas Reto 6: Caídas y comportamientos de riesgo
# =====================================================================
def test_reto6_clasificar_posturas_sin_falsos_positivos():
    detector = DetectorCaidasRiesgo(umbral_frames_no_se_levanta=10)

    # De pie
    p_pie = detector.clasificar_postura(w=64, h=180, altura_erguida_ref=180, frames_horizontal=0)
    assert p_pie["estado"] == "de_pie_o_caminando"
    assert p_pie["alerta"] is False

    # Sentado -> NO alerta
    p_sentado = detector.clasificar_postura(w=72, h=124, altura_erguida_ref=180, frames_horizontal=0)
    assert p_sentado["estado"] == "sentado"
    assert p_sentado["alerta"] is False

    # Agachado -> NO alerta
    p_agachado = detector.clasificar_postura(w=68, h=82, altura_erguida_ref=180, frames_horizontal=0)
    assert p_agachado["estado"] == "agachado"
    assert p_agachado["alerta"] is False

    # Caída confirmada (horizontal en suelo y no se levanta) -> ALERTA
    p_caida = detector.clasificar_postura(w=184, h=54, altura_erguida_ref=180, frames_horizontal=14)
    assert p_caida["estado"] == "caida_confirmada"
    assert p_caida["alerta"] is True


def test_reto6_video_y_endpoint():
    resp = client.post("/vision/reto6/caidas-riesgo")
    assert resp.status_code == 200
    data = resp.json()
    assert data["reto"] == "6. Caídas y comportamientos de riesgo"
    assert data["total_alertas_caida"] == 1
    assert data["conteo_posturas_observadas"]["agachado"] > 0
    assert data["conteo_posturas_observadas"]["sentado"] > 0
    assert data["conteo_posturas_observadas"]["caida_confirmada"] > 0


# =====================================================================
# Pruebas Reto 7: Correspondencia en casillero
# =====================================================================
def test_reto7_correspondencia_casilleros_y_reporte_apartamento():
    resp = client.post("/vision/reto7/correspondencia-casillero")
    assert resp.status_code == 200
    data = resp.json()

    assert data["reto"] == "7. Correspondencia en casillero"
    assert data["total_casilleros_ocupados"] == 4

    por_casillero = {
        item["casillero"]: item for item in data["correspondencia_detectada"]
    }
    assert por_casillero["C101"]["apartamento"] == 101
    assert por_casillero["C101"]["tipo_correspondencia"] == "paquete"
    assert "Diego López" in por_casillero["C101"]["residentes_notificados"]

    assert por_casillero["C103"]["apartamento"] == 103
    assert por_casillero["C103"]["tipo_correspondencia"] == "recibo"

    assert por_casillero["C202"]["apartamento"] == 202
    assert por_casillero["C202"]["tipo_correspondencia"] == "paquete"

    assert por_casillero["C204"]["apartamento"] == 204
    assert por_casillero["C204"]["tipo_correspondencia"] == "recibo"


def test_analizar_todos_los_retos_endpoint():
    resp = client.post("/vision/analizar-todos")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["retos"]) == 7
