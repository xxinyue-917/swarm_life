from fastapi.testclient import TestClient

from backend.api import app


def test_get_config_returns_matrix_and_config():
    client = TestClient(app)
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "config" in data
    assert "matrix" in data
    assert isinstance(data["matrix"], list)


def test_websocket_streams_state_messages():
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        message = websocket.receive_json()
        assert message["type"] == "state"
        payload = message["payload"]
        assert "particles" in payload
        assert "matrix" in payload


def test_post_config_resizes_matrix_with_reset():
    client = TestClient(app)
    response = client.get("/config")
    initial = response.json()
    current_species = initial["config"]["species_count"]
    new_species = current_species + 1

    response = client.post(
        "/config",
        json={"config": {"species_count": new_species}, "reset_matrix": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["species_count"] == new_species
    assert len(payload["matrix"]) == new_species
    for row in payload["matrix"]:
        assert len(row) == new_species
    client.post(
        "/config",
        json={"config": {"species_count": current_species}, "reset_matrix": True},
    )


def test_websocket_update_config_resets_matrix():
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()  # initial state
        websocket.send_json(
            {
                "type": "update_config",
                "config": {"species_count": 5},
                "reset_matrix": True,
            }
        )
        message = websocket.receive_json()
        assert message["type"] == "state"
        matrix = message["payload"]["matrix"]
        assert len(matrix) == 5
        for row in matrix:
            assert len(row) == 5
    client.post(
        "/config",
        json={"config": {"species_count": 3}, "reset_matrix": True},
    )
