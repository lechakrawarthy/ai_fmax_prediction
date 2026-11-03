import pytest
from fastapi.testclient import TestClient

from fmax import config as C
from fmax.data import load_raw

bundle = C.ARTIFACT_DIR / "model_bundle.joblib"
pytestmark = pytest.mark.skipif(not bundle.exists(), reason="run `python -m fmax.pipeline` first")


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_predict_and_simulate(client):
    row = load_raw().query("split == 'test'").iloc[0]
    unit = {c: float(row[c]) for c in C.FEATURES}
    p = client.post("/predict", json=[unit]).json()[0]
    assert p["action"] == "windowed_search"
    assert p["lower_bound_mhz"] < p["fmax_pred_mhz"] < p["upper_bound_mhz"]

    s = client.post("/simulate", json={"unit": unit, "fmax_true_mhz": row["fmax_true_mhz"]}).json()
    assert abs(s["measured_mhz"] - row["fmax_true_mhz"]) <= 20
    assert s["n_tests"] == len(s["trajectory"])


def test_ood_unit_triggers_fallback(client):
    unit = {c: 0.0 for c in C.FEATURES} | {"ro_01": 25.0}
    p = client.post("/predict", json=[unit]).json()[0]
    assert p["action"] == "fallback_full_search" and p["fmax_pred_mhz"] is None
    s = client.post("/simulate", json={"unit": unit, "fmax_true_mhz": 2000.0}).json()
    assert s["fallback"] is True and s["measured_mhz"] in (1980.0, 1990.0, 2000.0)
