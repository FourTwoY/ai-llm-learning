from lora_domain_learning_assistant.src.api.app import app


try:
    from fastapi.testclient import TestClient

    client = TestClient(app)
except Exception:
    class _Response:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            if hasattr(self._payload, "model_dump"):
                return self._payload.model_dump()
            return self._payload

    class _FallbackClient:
        def __init__(self, fastapi_app):
            self.app = fastapi_app

        def get(self, path: str):
            handler = self.app.routes[("GET", path)]
            return _Response(handler())

    client = _FallbackClient(app)


def test_health_api():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "model" in payload["data"]
