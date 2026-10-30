"""HTTP boundary and genuine research integration tests; all writes use temporary storage."""
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.main import create_app
from api.jobs import JobManager
from app.configuration.settings import AppSettings, SettingsManager
from experiments.store import ExperimentStore
from ml.datasets.loader import load_dataset


@pytest.fixture
def web(tmp_path, bc_experiment):
    storage = tmp_path / "storage"
    rec = bc_experiment["record"]
    shutil.copytree(bc_experiment["storage"] / rec["artifact_dir"], storage / rec["artifact_dir"])
    ExperimentStore(storage).save(rec)
    manager = SettingsManager(tmp_path / "settings.json")
    manager.save(AppSettings(storage_dir=str(storage), data_dir=str(Path("data").resolve())))
    app = create_app(manager.path)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, app, rec


def sign_in(client):
    result = client.post("/api/auth/register", json={"username":"webtester", "password":"test-only-password"})
    assert result.status_code == 200, result.text
    client.headers["X-CSRF-Token"] = result.json()["csrf"]
    return result


def test_public_health_and_protected_history(web):
    client, _, _ = web
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/experiments").status_code == 401
    assert client.get("/api/auth/session").json()["registration_open"]


def test_cookie_csrf_logout_and_registration_boundary(web):
    client, _, _ = web
    result = sign_in(client)
    cookie = result.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.get("/api/auth/session").json()["authenticated"]
    client.headers.pop("X-CSRF-Token")
    assert client.post("/api/federation", json={}).status_code == 403
    client.headers["X-CSRF-Token"] = result.json()["csrf"]
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/dashboard").status_code == 401
    assert client.post("/api/auth/register", json={"username":"intruder","password":"somepassword"}).status_code == 401


def test_existing_auth_login_reused(web):
    client, _, _ = web
    sign_in(client)
    client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={"username":"webtester", "password":"incorrect-password"}).status_code == 401
    assert client.post("/api/auth/login", json={"username":"webtester", "password":"test-only-password"}).status_code == 200


def test_origin_body_size_and_redacted_errors(web):
    client, _, _ = web
    body = {"username":"webtester", "password":"test-only-password"}
    assert client.post("/api/auth/register", json=body, headers={"origin":"https://evil.example"}).status_code == 403
    assert client.post("/api/auth/login", content="x" * 65537).status_code == 413
    result = client.post("/api/auth/login", json={"username":"aa", "password":"SECRET"})
    assert result.status_code == 422
    assert "SECRET" not in result.text and '"input"' not in result.text


def test_rate_limit(web):
    client, _, _ = web
    for _ in range(10):
        assert client.post("/api/auth/login", json={"username":"unknown", "password":"test-password"}).status_code == 401
    assert client.post("/api/auth/login", json={"username":"unknown", "password":"test-password"}).status_code == 429


@pytest.mark.parametrize("endpoint", ["/api/dashboard", "/api/datasets", "/api/settings", "/api/models", "/api/schema"])
def test_read_routes_and_response_security(web, endpoint):
    client, _, _ = web
    sign_in(client)
    response = client.get(endpoint)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "C:\\Users" not in response.text


def test_dataset_inventory_and_historical_pagination(web):
    client, _, rec = web
    sign_in(client)
    datasets = client.get("/api/datasets").json()
    assert len(datasets) == 5
    # CSV datasets are downloaded separately (data/README.md); every dataset that is present must validate.
    assert all(d["status"].startswith("valid") for d in datasets if d["available"])
    assert any(d["key"] == "breast_cancer" and d["available"] for d in datasets)  # bundled with scikit-learn
    result = client.get("/api/experiments", params={"q":rec["id"],"page_size":1}).json()
    assert result["total"] == 1 and result["items"][0]["id"] == rec["id"]
    assert result["items"][0]["origin"] == "historical"
    assert client.get("/api/experiments", params={"page":2,"page_size":1}).json()["items"] == []


@pytest.mark.parametrize("patch", [{"num_clients":1}, {"num_rounds":501}, {"dataset_key":"unknown"}, {"seed":-1},
    {"malicious_clients":[30],"attack_type":"sign_flip"}, {"attack_type":"sign_flip","malicious_clients":[]},
    {"trust_alpha":1}, {"learning_rate":"NaN"}, {"storage_dir":"C:/"}])
def test_training_rejects_invalid_and_extra_config(web, patch):
    client, _, _ = web
    sign_in(client)
    response = client.post("/api/federation", json={"dataset_key":"breast_cancer", **patch})
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("identifier", ["..%5Cusers", "bad.id", "models%5Csecret"])
def test_path_traversal_is_rejected(web, identifier):
    client, _, _ = web
    sign_in(client)
    assert client.get(f"/api/experiments/{identifier}").status_code in {400,404}


def test_prediction_uses_real_artifact_and_shap(web):
    client, _, rec = web
    sign_in(client)
    values = load_dataset("breast_cancer", Path("data")).X.iloc[0].to_dict()
    response = client.post("/api/predictions", json={"experiment_id":rec["id"],"values":values})
    assert response.status_code == 200, response.text
    out = response.json()
    assert 0 <= out["probability_positive"] <= 1
    explanation = out["explanation"]
    assert explanation and out["explanation_error"] is None
    assert abs(explanation["base_value"] + sum(c["shap"] for c in explanation["contributions"]) - explanation["output_value"]) < 1e-6
    assert out["experiment_id"] == rec["id"]
    assert client.post("/api/predictions", json={"experiment_id":rec["id"],"values":{}}).status_code == 422


def test_compare_requires_identical_evaluation_protocol(web):
    client, app, rec = web
    sign_in(client)
    other = json.loads(json.dumps(rec))
    other["id"] += "-comparison"
    app.state.research.store.save(other)
    response = client.post("/api/experiments/compare", json={"ids":[rec["id"],other["id"]]})
    assert response.status_code == 200
    other["split_config"]["seed"] += 1
    app.state.research.store.save(other)
    assert client.post("/api/experiments/compare", json={"ids":[rec["id"],other["id"]]}).status_code == 409


@pytest.mark.parametrize("format", ["json", "html", "pdf"])
def test_reports_from_existing_engine(web, format):
    client, app, rec = web
    sign_in(client)
    response = client.get(f"/api/reports/{rec['id']}?format={format}")
    assert response.status_code == 200, response.text[:200]
    assert rec["id"] in response.headers["content-disposition"]
    assert not list((app.state.research.storage / "web/exports").glob("*"))


def test_genuine_web_simulation_persists_callbacks_and_preserves_history(web):
    client, app, rec = web
    sign_in(client)
    original = app.state.research.store.dir / f"{rec['id']}.json"
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    response = client.post("/api/federation", json={"dataset_key":"breast_cancer","num_clients":3,"num_rounds":3,"local_epochs":1,"malicious_clients":[3],"attack_type":"sign_flip"})
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get(f"/api/federation/{job_id}").json()
        if job["status"] != "running": break
        time.sleep(.05)
    assert job["status"] == "completed", job
    assert job["completed_rounds"] == 3 and job["experiment_id"]
    kinds = {e["type"] for e in job["events"]}
    assert {"client_training","client_submitted","aggregating","evaluating"} <= kinds
    for rnd in job["rounds"]:
        if rnd["model_updated"]:
            assert abs(sum(c["weight"] for c in rnd["clients"]) - 1) < 1e-8
    assert client.get(f"/api/experiments/{job['experiment_id']}").json()["origin"] == "web"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    assert (app.state.jobs.directory / f"{job_id}.json").is_file()
    listing = client.get("/api/federation")
    assert len(listing.content) < 4000
    assert listing.json()[0]["id"] == job_id


def test_conflict_cancellation_and_recovery(tmp_path, monkeypatch):
    entered = threading.Event()
    def controlled_run(config, data_dir, storage_dir, on_event, stop_event):
        entered.set()
        stop_event.wait(5)
        from federated.server.runner import ExperimentCancelled
        raise ExperimentCancelled("cooperative stop before first round")
    monkeypatch.setattr("api.jobs.run_experiment", controlled_run)
    manager = JobManager(Path("data"), tmp_path)
    job = manager.start({"dataset_key":"breast_cancer"})
    assert entered.wait(2)
    with pytest.raises(RuntimeError): manager.start({"dataset_key":"heart"})
    manager.cancel(job["id"])
    manager.close()
    assert manager.get(job["id"])["status"] == "cancelled"
    raw = manager.get(job["id"])
    raw["status"] = "running"
    (manager.directory / f"{job['id']}.json").write_text(json.dumps(raw))
    recovered = JobManager(Path("data"), tmp_path)
    assert recovered.get(job["id"])["status"] == "interrupted"
    recovered.close()


def test_defaults_are_separate_from_desktop_settings(web):
    client, app, _ = web
    sign_in(client)
    settings = client.get("/api/settings").json()["defaults"]
    settings["num_rounds"] = 9
    assert client.put("/api/settings", json={"defaults":settings}).status_code == 200
    assert client.get("/api/settings").json()["defaults"]["num_rounds"] == 9
    assert app.state.research.settings.num_rounds == 30


def test_artifact_path_escape_is_rejected_before_deserialization(web):
    client, app, rec = web
    sign_in(client)
    rec = json.loads(json.dumps(rec))
    rec["artifact_dir"] = "../../outside"
    app.state.research.store.save(rec)
    response = client.post("/api/predictions", json={"experiment_id":rec["id"],"values":{}})
    assert response.status_code == 400


def test_expired_session_is_rejected(web):
    client, app, _ = web
    sign_in(client)
    for session in app.state.sessions.sessions.values():
        session["expires"] = 0
    assert client.get("/api/models").status_code == 401
    assert not client.get("/api/auth/session").json()["authenticated"]


def test_failure_during_cancellation_is_not_mislabelled(tmp_path, monkeypatch):
    entered = threading.Event()
    def failing(config, data_dir, storage_dir, on_event, stop_event):
        entered.set()
        stop_event.wait(5)
        raise RuntimeError("an unrelated failure")
    monkeypatch.setattr("api.jobs.run_experiment", failing)
    manager = JobManager(Path("data"), tmp_path)
    job = manager.start({"dataset_key":"breast_cancer"})
    assert entered.wait(2)
    manager.cancel(job["id"])
    manager.close()
    assert manager.get(job["id"])["status"] == "failed"
