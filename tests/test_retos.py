"""Pruebas de los motores de los retos 2 a 7."""

import numpy as np
import pytest

from app.vision.conteo import LINEAS, lado
from app.vision.placas import normalizar_placa, votar


# ---------------------------------------------------------------- funciones puras

def test_lado_de_linea_con_margen():
    linea = LINEAS[0]  # horizontal al 80 % de la altura
    assert lado(linea, (100, 300), 768, 432, 6) == -1   # arriba: lado de entrada
    assert lado(linea, (100, 400), 768, 432, 6) == 1    # abajo
    assert lado(linea, (100, 346), 768, 432, 6) == 0    # dentro del margen: no decide


@pytest.mark.parametrize("texto, esperado", [
    ("KXw694", ("KXW694", "carro")),
    ("MTK48D", ("MTK48D", "moto")),       # no se confunde con MTK480
    ("WGB 442", ("WGB442", "carro")),
    ("W6B442", ("WGB442", "carro")),      # 6 en posición de letra → G
    ("KXWG94", ("KXW694", "carro")),      # G en posición de número → 6
    ("HOLA", None),
    ("1234567", None),
])
def test_normalizar_placa(texto, esperado):
    assert normalizar_placa(texto) == esperado


def test_votacion_corrige_lecturas_aisladas():
    lecturas = [("KXW694", "carro", 0.9), ("KXW694", "carro", 0.8), ("KXV694", "carro", 0.5)]
    assert votar(lecturas) == ("KXW694", "carro")


def test_distancia_sobre_el_piso_relativa_a_la_altura():
    from app.vision.abandono import distancia_relativa

    objeto = [100, 300, 40, 60]                       # base en y = 360
    al_lado = [150, 160, 60, 200]                     # pies en (180, 360)
    al_fondo = [150, 50, 30, 100]                     # pies en (165, 150), persona pequeña
    assert distancia_relativa(objeto, al_lado) < 0.3
    assert distancia_relativa(objeto, al_fondo) > 1.5


def test_clasificar_correspondencia():
    import cv2

    from app.vision.casilleros import clasificar

    recorte = np.zeros((100, 100, 3), np.uint8)
    mascara = np.zeros((100, 100), np.uint8)
    recorte[30:80, 20:80] = (40, 110, 180)            # caja de cartón (BGR)
    mascara[30:80, 20:80] = 1
    assert clasificar(recorte, mascara)[0] == "paquete"

    recorte[:] = 0
    mascara[:] = 0
    recorte[70:80, 15:85] = (235, 235, 235)           # sobre blanco y plano
    mascara[70:80, 15:85] = 1
    assert clasificar(recorte, mascara)[0] == "recibo"
    assert cv2 is not None


# ---------------------------------------------------------------- videos de prueba

def test_reto2_conteo_entradas_salidas():
    pytest.importorskip("ultralytics")
    from app.vision.conteo import ContadorFlujo

    r = ContadorFlujo().analizar_video("data/videos/reto2_conteo_flujo.mp4")["summary"]
    assert r["total_entradas"] == 6
    assert r["total_salidas"] == 3
    assert r["max_personas_simultaneas"] >= 3 and r["hubo_hora_pico"]
    assert all(e["sentido"] in ("entrada", "salida") for e in r["eventos_cruce"])


def test_reto3_merodeo_sin_alertar_conversacion():
    pytest.importorskip("ultralytics")
    from app.vision.merodeo import DetectorMerodeo

    r = DetectorMerodeo().analizar_video("data/videos/reto3_merodeo.mp4")["summary"]
    clasif = {p["person_id"]: p["clasificacion"] for p in r["personas"]}
    assert r["total_alertas_merodeo"] >= 2
    assert list(clasif.values()).count("conversando") == 2          # la pareja del final
    assert any(p["regresos"] >= 2 and p["alerta"] for p in r["personas"])  # pasó 3 veces
    assert any(c == "paso_normal" for c in clasif.values())          # quien solo pasa


def test_reto4_objeto_abandonado():
    pytest.importorskip("ultralytics")
    from app.vision.abandono import DetectorAbandono

    r = DetectorAbandono().analizar_video("data/videos/reto4_objeto_abandonado.mp4")["summary"]
    assert r["total_alertas_abandono"] == 1
    alerta = r["alertas"][0]
    assert alerta["tipo_objeto"] in ("morral", "bolso", "maleta")
    assert alerta["propietario_id"] is not None
    assert 150 <= alerta["frame_alerta"] <= 200      # unos 3 s después de que el dueño se va


def test_reto5_una_lectura_por_vehiculo_dia_y_noche():
    pytest.importorskip("rapidocr_onnxruntime")
    from app.vision.placas import LectorPlacas

    r = LectorPlacas().analizar_video("data/videos/reto5_lectura_placas.mp4")["summary"]
    placas = {v["placa"]: v for v in r["lecturas_unicas"]}
    assert set(placas) == {"KXW694", "MTK48D", "WGB442"}
    assert placas["MTK48D"]["tipo_vehiculo"] == "moto"
    assert placas["WGB442"]["condicion_luz"] == "noche"
    assert placas["KXW694"]["autorizado"] and placas["KXW694"]["apartamento"] == 101
    assert not placas["MTK48D"]["autorizado"]


def test_reto6_caida_sin_confundir_sentarse_o_agacharse():
    from app.vision.caidas import DetectorCaidas

    r = DetectorCaidas().analizar_video("data/videos/reto6_caidas_riesgo.mp4")["summary"]
    assert r["total_alertas_caida"] == 1
    posturas = {p["person_id"]: p for p in r["personas"]}
    assert "agachado" in posturas[1]["posturas"] and not posturas[1]["cayo"]
    assert "sentado" in posturas[2]["posturas"] and not posturas[2]["cayo"]
    assert posturas[3]["cayo"]


def test_reto7_casilleros_con_numero_de_apartamento():
    from app.vision.casilleros import MonitorCasilleros

    r = MonitorCasilleros().analizar_video("data/videos/reto7_casilleros.mp4")["summary"]
    assert r["total_casilleros"] == 8
    pendientes = {c["casillero"]: c for c in r["correspondencia_detectada"]}
    assert {k: c["tipo_correspondencia"] for k, c in pendientes.items()} == {
        "C101": "paquete", "C103": "recibo", "C202": "paquete", "C204": "recibo",
    }
    assert pendientes["C101"]["apartamento"] == 101
    assert pendientes["C101"]["residentes_notificados"]
