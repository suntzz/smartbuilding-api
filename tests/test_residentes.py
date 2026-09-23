from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
'''def test_get_residente_por_id():
    response = client.get("/residentes/1")
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "full_name": "Juan Perez",
        "tower": 1,
        "apartment": 101
    }

def test_get_residente_csv():
    response = client.get("/residentes/10")
    assert response.status_code == 200
    assert response.json() == {
        "id": 10,
        "full_name": "Óscar Acosta",
        "tower": 1,
        "apartment": 103
    }
'''
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