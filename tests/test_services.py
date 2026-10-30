import json

import pytest

from app.configuration.settings import AppSettings, SettingsError, SettingsManager
from app.services.auth import AuthError, LocalAuth
from app.services.context import AppContext
from app.services.jobs import CancelToken, run_job
from app.services.reports import ReportError, ReportOptions, generate_report
from experiments.store import ExperimentStore
from ml.models.artifact import ArtifactError


# ------------------------------------------------------------------ settings
def test_settings_roundtrip(tmp_path):
    m = SettingsManager(tmp_path / "settings.json")
    s = AppSettings(num_rounds=12, learning_rate=0.05, malicious_clients=[2, 3])
    m.save(s)
    loaded = m.load()
    assert loaded == s and not m.warnings


def test_invalid_settings_rejected_on_save(tmp_path):
    m = SettingsManager(tmp_path / "settings.json")
    with pytest.raises(SettingsError):
        m.save(AppSettings(num_clients=1))
    with pytest.raises(SettingsError):
        m.save(AppSettings(malicious_clients=[7]))
    assert not (tmp_path / "settings.json").exists()


def test_corrupt_settings_fall_back_to_defaults(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{not json", encoding="utf-8")
    m = SettingsManager(p)
    assert m.load() == AppSettings()
    assert m.warnings and (tmp_path / "settings.invalid.json").exists()
    p.write_text(json.dumps({"num_rounds": "many"}), encoding="utf-8")
    assert m.load() == AppSettings() and m.warnings


def test_settings_to_experiment_config():
    s = AppSettings(trust_aware_enabled=False, baseline_strategy="median",
                    malicious_simulation_enabled=False)
    cfg = s.experiment_config()
    assert cfg.strategy == "median" and cfg.malicious_clients == [] and cfg.attack_type == "none"


# ------------------------------------------------------------------ auth
def test_local_auth(tmp_path):
    auth = LocalAuth(tmp_path)
    assert not auth.has_users()
    with pytest.raises(AuthError):
        auth.create_user("ab", "password123")
    with pytest.raises(AuthError):
        auth.create_user("alice", "short")
    auth.create_user("alice", "correct horse")
    assert auth.has_users()
    assert auth.verify("alice", "correct horse")
    assert not auth.verify("alice", "wrong password")
    assert not auth.verify("bob", "correct horse")
    raw = (tmp_path / "users.json").read_text()
    assert "correct horse" not in raw
    auth.reset()
    assert not auth.has_users()


# ------------------------------------------------------------------ jobs
def test_run_job_captures_errors():
    ok = run_job(lambda a, b: a + b, 2, 3)
    assert ok.ok and ok.result == 5
    def fail():
        raise RuntimeError("boom")
    bad = run_job(fail)
    assert not bad.ok and "RuntimeError: boom" in bad.error and "Traceback" in bad.details
    t = CancelToken()
    assert not t.cancelled
    t.cancel()
    assert t.cancelled


# ------------------------------------------------------------------ store & context
def test_store_lists_and_skips_corrupt(bc_experiment):
    store = ExperimentStore(bc_experiment["storage"])
    (store.dir / "garbage.json").write_text("{", encoding="utf-8")
    recs = store.list()
    assert any(r["id"] == bc_experiment["record"]["id"] for r in recs)
    assert store.latest("breast_cancer")["artifact_dir"]
    assert store.latest("heart") is None


def test_context_finds_model_only_for_matching_dataset(bc_experiment, tmp_path):
    m = SettingsManager(tmp_path / "s.json")
    m.save(AppSettings(storage_dir=str(bc_experiment["storage"])))
    ctx = AppContext(m)
    art = ctx.artifact_for("breast_cancer")
    assert art.dataset_key == "breast_cancer"
    with pytest.raises(ArtifactError, match="No trained model"):
        ctx.artifact_for("heart")
    seen = []
    ctx.on("reference_mode_changed", seen.append)
    ctx.set_reference_mode(True)
    assert seen == [True] and ctx.reference_mode


# ------------------------------------------------------------------ reports
@pytest.mark.parametrize("fmt", ["pdf", "html", "json"])
def test_report_generation_uses_real_record(bc_experiment, tmp_path, fmt):
    rec = bc_experiment["record"]
    importance = [("Worst Radius", 0.5), ("Mean Texture", 0.2)]
    out = generate_report(rec, tmp_path / f"r.{fmt}", fmt, ReportOptions(), importance=importance)
    assert out.exists() and out.stat().st_size > 500
    if fmt != "pdf":
        text = out.read_text(encoding="utf-8")
        assert rec["id"] in text
        assert f"{rec['test_metrics']['roc_auc']:.3f}" in text
        assert "test" in text.lower()
    else:
        assert out.read_bytes()[:4] == b"%PDF"


def test_report_refuses_failed_or_running_experiment(tmp_path):
    with pytest.raises(ReportError):
        generate_report({"id": "x", "status": "failed"}, tmp_path / "r.pdf", "pdf", ReportOptions())
    with pytest.raises(ReportError):
        generate_report({"id": "x", "status": "completed"}, tmp_path / "r.doc", "doc", ReportOptions())


def test_report_distinguishes_splits_and_states_limitations(bc_experiment, tmp_path):
    rec = bc_experiment["record"]
    text = generate_report(rec, tmp_path / "r.html", "html", ReportOptions()).read_text(encoding="utf-8")
    assert "Held-out test split" in text and "Validation split" in text and "NOT test" in text
    assert "Limitations" in text and "not a deployed" in text
    assert "Reproducibility" in text and "validation/test rows not used" in text
    assert "client_" not in text.split("<h2>Trust analysis</h2>")[1].split("<h2>")[0]


def test_report_refuses_superseded_experiment(bc_experiment, tmp_path):
    stale = dict(bc_experiment["record"])
    stale.pop("pipeline_version")
    with pytest.raises(ReportError, match="superseded"):
        generate_report(stale, tmp_path / "r.html", "html", ReportOptions())
