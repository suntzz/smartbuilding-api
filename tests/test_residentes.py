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
'''
def test_get_residente_csv():
    response = client.get("/residentes/10")
    assert response.status_code == 200
    assert response.json() == {
        "id": 10,
        "full_name": "Óscar Acosta",
        "tower": 1,
        "apartment": 103
    }