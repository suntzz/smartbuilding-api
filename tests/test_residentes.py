from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_salud_endpoint_y_cabeceras_seguridad():
    response = client.get("/salud")
    assert response.status_code == 200
    data = response.json()
    assert data["estado"] == "ok"
    assert data["residentes_cargados"] > 0
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"


def test_get_residente_con_campos_ampliados():
    response = client.get("/residentes/10")
    assert response.status_code == 200
    data = response.json()
    assert data["tipo"] == "residente"
    assert data["telefono"] == "3000000010"
    assert data["email"] == "oscar.acosta.10@example.com"


def test_get_residente_no_existente():
    response = client.get("/residentes/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Residente no encontrado"}


def test_filtrar_residentes_por_busqueda_y_torre():
    res_q = client.get("/residentes", params={"q": "acosta"})
    assert res_q.status_code == 200
    lista_q = res_q.json()
    assert len(lista_q) >= 1
    assert any("Acosta" in r["full_name"] for r in lista_q)

    res_torre = client.get("/residentes", params={"torre": 1})
    assert res_torre.status_code == 200
    lista_torre = res_torre.json()
    assert len(lista_torre) > 0
    assert all(r["tower"] == 1 for r in lista_torre)


def test_analizar_imagen_validacion_bordes():
    res_vacio = client.post(
        "/vision/demografia-vestimenta/analizar-imagen",
        files={"archivo": ("vacio.jpg", b"", "image/jpeg")},
    )
    assert res_vacio.status_code == 400
    assert "vacío" in res_vacio.json()["detail"]

    res_invalido = client.post(
        "/vision/demografia-vestimenta/analizar-imagen",
        files={"archivo": ("corrupto.jpg", b"no-es-una-imagen", "image/jpeg")},
    )
    assert res_invalido.status_code == 400
    assert "inválido" in res_invalido.json()["detail"]
