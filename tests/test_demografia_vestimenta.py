import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.vision.demografia_vestimenta import (
    COLORES_VALIDOS,
    RANGOS_EDAD,
    SEXOS_APARENTES,
    TIPOS_PRENDA_INFERIOR,
    TIPOS_PRENDA_SUPERIOR,
    DetectorDemografiaVestimenta,
    analizar_recorte_persona,
    clasificar_color_bgr,
    clasificar_tipo_prenda_inferior,
    clasificar_tipo_prenda_superior,
    estimar_demografia_persona,
)

client = TestClient(app)


def _crear_parche_bgr(b: int, g: int, r: int, alto: int = 60, ancho: int = 60) -> np.ndarray:
    img = np.zeros((alto, ancho, 3), dtype=np.uint8)
    img[:] = (b, g, r)
    return img


def _crear_persona_sintetica(
    color_sup_bgr=(20, 20, 200),
    color_inf_bgr=(160, 70, 25),
    con_mangas_cortas=True,
    es_falda=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera una imagen BGR y máscara binaria de una persona sintética."""
    h, w = 220, 90
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Cabeza y cabello (0% - 20%)
    img[8:42, 28:62] = (40, 50, 70)
    mask[8:42, 28:62] = 255

    # Torso (21% - 54%)
    img[46:118, 18:72] = color_sup_bgr
    mask[46:118, 18:72] = 255
    if con_mangas_cortas:
        # Brazos laterales con tono de piel
        img[62:114, 14:26] = (130, 165, 215)
        img[62:114, 64:76] = (130, 165, 215)
        mask[62:114, 14:76] = 255

    # Piernas / prenda inferior (54% - 90%)
    if es_falda:
        # Falda ancha arriba (118..162) y tobillos delgados abajo (162..198)
        img[118:162, 16:74] = color_inf_bgr
        mask[118:162, 16:74] = 255
        img[162:198, 38:52] = (130, 165, 215)
        mask[162:198, 38:52] = 255
    else:
        # Dos columnas de pantalón separadas en el centro
        img[118:198, 22:42] = color_inf_bgr
        img[118:198, 48:68] = color_inf_bgr
        mask[118:198, 22:42] = 255
        mask[118:198, 48:68] = 255

    return img, mask


def test_clasificar_color_bgr_basicos():
    assert clasificar_color_bgr(_crear_parche_bgr(15, 15, 15))["color"] == "negro"
    assert clasificar_color_bgr(_crear_parche_bgr(245, 245, 245))["color"] == "blanco"
    assert clasificar_color_bgr(_crear_parche_bgr(120, 120, 120))["color"] == "gris"
    assert clasificar_color_bgr(_crear_parche_bgr(20, 20, 210))["color"] == "rojo"
    assert clasificar_color_bgr(_crear_parche_bgr(210, 80, 20))["color"] == "azul"
    assert clasificar_color_bgr(_crear_parche_bgr(30, 180, 30))["color"] == "verde"
    assert clasificar_color_bgr(_crear_parche_bgr(20, 220, 230))["color"] == "amarillo"


def test_clasificar_color_region_vacia_lanza_error():
    with pytest.raises(ValueError):
        clasificar_color_bgr(np.array([], dtype=np.uint8))


def test_clasificar_tipo_prenda_superior_e_inferior():
    img_camiseta, mask_camiseta = _crear_persona_sintetica(
        color_sup_bgr=(25, 25, 200),
        color_inf_bgr=(165, 75, 20),
        con_mangas_cortas=True,
        es_falda=False,
    )
    res_1 = analizar_recorte_persona(img_camiseta, mask_camiseta, person_id=1)
    assert res_1["prenda_superior"]["color"] == "rojo"
    assert res_1["prenda_superior"]["tipo"] in {"camiseta", "blusa"}
    assert res_1["prenda_inferior"]["color"] in {"azul", "azul oscuro"}
    assert res_1["prenda_inferior"]["tipo"] == "jean"

    img_falda, mask_falda = _crear_persona_sintetica(
        color_sup_bgr=(200, 100, 30),
        color_inf_bgr=(230, 230, 230),
        con_mangas_cortas=False,
        es_falda=True,
    )
    res_2 = analizar_recorte_persona(img_falda, mask_falda, person_id=2)
    assert res_2["prenda_inferior"]["tipo"] == "falda"
    assert res_2["prenda_inferior"]["color"] == "blanco"
    assert res_2["sexo_aparente"] == "Femenino"


def test_validaciones_recorte_invalido():
    with pytest.raises(ValueError):
        analizar_recorte_persona(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        clasificar_tipo_prenda_superior(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        clasificar_tipo_prenda_inferior(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        estimar_demografia_persona(np.array([], dtype=np.uint8))


def test_detector_analizar_video_real():
    detector = DetectorDemografiaVestimenta()
    reporte = detector.analizar_video(
        "data/videos/reto1_demografia_vestimenta.mp4",
        cada_n_frames=3,
        min_frames_persona=3,
    )

    assert reporte["reto"] == "1. Demografía y vestimenta"
    assert reporte["total_personas_detectadas"] >= 4
    for persona in reporte["personas"]:
        assert persona["sexo_aparente"] in SEXOS_APARENTES
        assert persona["rango_edad"] in RANGOS_EDAD
        assert persona["prenda_superior"]["tipo"] in TIPOS_PRENDA_SUPERIOR
        assert persona["prenda_superior"]["color"] in COLORES_VALIDOS
        assert persona["prenda_inferior"]["tipo"] in TIPOS_PRENDA_INFERIOR
        assert persona["prenda_inferior"]["color"] in COLORES_VALIDOS


def test_detector_video_inexistente():
    detector = DetectorDemografiaVestimenta()
    with pytest.raises(FileNotFoundError):
        detector.analizar_video("data/videos/no_existe.mp4")


def test_endpoints_api_demografia_vestimenta():
    resp_videos = client.get("/vision/videos")
    assert resp_videos.status_code == 200
    nombres = [v["nombre"] for v in resp_videos.json()["videos_locales"]]
    assert "reto1_demografia_vestimenta.mp4" in nombres

    resp_analisis = client.post(
        "/vision/demografia-vestimenta/analizar-video?video=reto1_demografia_vestimenta.mp4&cada_n_frames=4"
    )
    assert resp_analisis.status_code == 200
    data = resp_analisis.json()
    assert data["total_personas_detectadas"] >= 4
    assert "por_sexo_aparente" in data["resumen_demografico"]

    resp_404 = client.post(
        "/vision/demografia-vestimenta/analizar-video?video=video_inexistente.mp4"
    )
    assert resp_404.status_code == 404

    resp_eventos = client.get("/vision/demografia-vestimenta/eventos")
    assert resp_eventos.status_code == 200
    assert resp_eventos.json()["total_analisis"] >= 1

    resp_dash = client.get("/dashboard")
    assert resp_dash.status_code == 200
    assert "Demografía y Vestimenta" in resp_dash.text


def test_endpoint_analizar_imagen_subida():
    escena = np.full((260, 320, 3), 240, dtype=np.uint8)
    persona, _ = _crear_persona_sintetica()
    escena[20:240, 115:205] = persona
    ok, buf = cv2.imencode(".png", escena)
    assert ok

    resp = client.post(
        "/vision/demografia-vestimenta/analizar-imagen",
        files={"archivo": ("test_persona.png", buf.tobytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["personas_en_frame"] >= 1
    p = body["personas"][0]
    assert p["sexo_aparente"] in SEXOS_APARENTES
    assert p["rango_edad"] in RANGOS_EDAD
