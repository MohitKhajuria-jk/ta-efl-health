"""Application settings: validated, persisted atomically as JSON."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from experiments.store import atomic_write_text
from federated.clients.attacks import ATTACK_DESCRIPTIONS
from federated.server.runner import ExperimentConfig, ExperimentError
from ml.datasets.registry import DatasetRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"


class SettingsError(ValueError):
    pass


@dataclass
class AppSettings:
    # dataset & paths
    default_dataset: str = "heart"
    data_dir: str = "data"
    storage_dir: str = "storage"
    # training parameters (defaults mirror the design reference)
    num_clients: int = 5
    num_rounds: int = 30
    batch_size: int = 32
    learning_rate: float = 0.01
    local_epochs: int = 5
    l2: float = 1e-4
    class_weight: str = "none"
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    fraction_fit: float = 1.0
    seed: int = 42
    decision_threshold: float = 0.5
    # aggregation & trust
    trust_aware_enabled: bool = True
    baseline_strategy: str = "fedavg"  # used when trust-aware aggregation is disabled
    trust_alpha: float = 0.7
    trust_initial: float = 0.5
    trust_performance_weight: float = 0.6
    trust_exclusion_threshold: float = 0.3
    trust_performance_mode: str = "relative"
    # robustness experiments
    malicious_simulation_enabled: bool = True
    malicious_clients: list[int] = field(default_factory=lambda: [4])
    attack_type: str = "sign_flip"
    attack_scale: float = 3.0
    attack_noise_std: float = 1.0
    # logging & output
    save_experiment_logs: bool = True
    generate_charts: bool = True
    log_level: str = "INFO"
    show_reference_values: bool = False
    remembered_username: str = ""

    # ------------------------------------------------------------------ paths
    def resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def data_path(self) -> Path:
        return self.resolve(self.data_dir)

    @property
    def storage_path(self) -> Path:
        return self.resolve(self.storage_dir)

    # ------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.default_dataset not in DatasetRegistry.keys():
            errors.append(f"Unknown dataset '{self.default_dataset}'")
        if self.attack_type not in ATTACK_DESCRIPTIONS:
            errors.append(f"Unknown attack type '{self.attack_type}'")
        if self.baseline_strategy not in ("fedavg", "median"):
            errors.append("Baseline strategy must be 'fedavg' or 'median'")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            errors.append("Log level must be DEBUG, INFO, WARNING or ERROR")
        for name in ("data_dir", "storage_dir"):
            if not str(getattr(self, name)).strip():
                errors.append(f"{name} must not be empty")
        try:
            self.experiment_config().validate()
        except ExperimentError as exc:
            errors.extend(str(exc).split("; "))
        return errors

    def experiment_config(self, **overrides: Any) -> ExperimentConfig:
        malicious = list(self.malicious_clients) if self.malicious_simulation_enabled else []
        cfg = ExperimentConfig(
            dataset_key=self.default_dataset, num_clients=self.num_clients,
            num_rounds=self.num_rounds, learning_rate=self.learning_rate,
            batch_size=self.batch_size, local_epochs=self.local_epochs, l2=self.l2,
            class_weight=self.class_weight,
            strategy="trust" if self.trust_aware_enabled else self.baseline_strategy,
            trust_alpha=self.trust_alpha, trust_initial=self.trust_initial,
            trust_performance_weight=self.trust_performance_weight,
            trust_exclusion_threshold=self.trust_exclusion_threshold,
            trust_performance_mode=self.trust_performance_mode,
            partition=self.partition, dirichlet_alpha=self.dirichlet_alpha,
            fraction_fit=self.fraction_fit, malicious_clients=malicious,
            attack_type=self.attack_type if malicious else "none",
            attack_scale=self.attack_scale, attack_noise_std=self.attack_noise_std,
            seed=self.seed, decision_threshold=self.decision_threshold,
        )
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise KeyError(k)
            setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AppSettings":
        """Build settings from a dict, coercing types and ignoring unknown keys."""
        base = AppSettings()
        kwargs: dict[str, Any] = {}
        for f in fields(AppSettings):
            if f.name not in d:
                continue
            default = getattr(base, f.name)
            v = d[f.name]
            try:
                if isinstance(default, bool):
                    if not isinstance(v, bool):
                        raise TypeError
                elif isinstance(default, int):
                    if isinstance(v, bool) or float(v) != int(float(v)):
                        raise TypeError
                    v = int(float(v))
                elif isinstance(default, float):
                    v = float(v)
                elif isinstance(default, str):
                    v = str(v)
                elif isinstance(default, list):
                    v = [int(x) for x in v]
            except (TypeError, ValueError) as exc:
                raise SettingsError(f"Setting '{f.name}' has an invalid value: {v!r}") from exc
            kwargs[f.name] = v
        return AppSettings(**kwargs)


class SettingsManager:
    """Loads and saves settings; a corrupt file is backed up and defaults are used."""

    def __init__(self, path: Path = DEFAULT_SETTINGS_PATH) -> None:
        self.path = Path(path)
        self.warnings: list[str] = []

    def load(self) -> AppSettings:
        self.warnings = []
        if not self.path.is_file():
            return AppSettings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise SettingsError("settings file is not a JSON object")
            s = AppSettings.from_dict(data)
            errs = s.validate()
            if errs:
                raise SettingsError("; ".join(errs))
            return s
        except (OSError, json.JSONDecodeError, SettingsError) as exc:
            backup = self.path.with_suffix(".invalid.json")
            try:
                shutil.copyfile(self.path, backup)
            except OSError:
                pass
            self.warnings.append(f"Settings file was invalid ({exc}); defaults loaded. "
                                 f"A copy was kept at {backup.name}.")
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        errs = settings.validate()
        if errs:
            raise SettingsError("; ".join(errs))
        atomic_write_text(self.path, json.dumps(settings.to_dict(), indent=2))
