"""Validated public contracts; research record payloads remain forward compatible."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from federated.server.runner import ExperimentConfig, ExperimentError
from ml.datasets.registry import DatasetRegistry


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Credentials(StrictModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class SessionInfo(BaseModel):
    authenticated: bool
    registration_open: bool = False
    username: str | None = None
    csrf: str | None = None


class TrainingRequest(StrictModel):
    dataset_key: str = "heart"
    strategy: Literal["trust", "fedavg", "median"] = "trust"
    num_clients: int = Field(default=5, ge=2, le=20)
    num_rounds: int = Field(default=30, ge=1, le=500)
    local_epochs: int = Field(default=5, ge=1, le=50)
    learning_rate: float = Field(default=.01, ge=.00001, le=1)
    batch_size: int = Field(default=32, ge=1, le=4096)
    l2: float = Field(default=.0001, ge=0, le=1)
    class_weight: Literal["none", "balanced"] = "none"
    partition: Literal["iid", "dirichlet"] = "iid"
    dirichlet_alpha: float = Field(default=.5, ge=.01, le=100)
    fraction_fit: float = Field(default=1, ge=.1, le=1)
    seed: int = Field(default=42, ge=0, le=2147483647)
    decision_threshold: float = Field(default=.5, ge=.05, le=.95)
    trust_alpha: float = Field(default=.7, ge=0, lt=1)
    trust_initial: float = Field(default=.5, ge=0, le=1)
    trust_performance_weight: float = Field(default=.6, ge=0, le=1)
    trust_exclusion_threshold: float = Field(default=.3, ge=0, le=1)
    trust_performance_mode: Literal["relative", "absolute"] = "relative"
    malicious_clients: list[int] = Field(default_factory=list, max_length=19)
    attack_type: Literal["none", "sign_flip", "scaling", "gaussian_noise", "label_flip", "invalid_update"] = "none"
    attack_scale: float = Field(default=3, ge=.01, le=100)
    attack_noise_std: float = Field(default=1, ge=0, le=100)
    label: str = Field(default="", max_length=100)

    @model_validator(mode="after")
    def validate_engine(self):
        if self.dataset_key not in DatasetRegistry.keys():
            raise ValueError("Unknown dataset")
        if len(set(self.malicious_clients)) != len(self.malicious_clients):
            raise ValueError("Attacking clients must be unique")
        if self.attack_type != "none" and not self.malicious_clients:
            raise ValueError("Choose at least one attacking client")
        try:
            ExperimentConfig(**self.model_dump()).validate()
        except ExperimentError as exc:
            raise ValueError(str(exc)) from None
        return self


class PredictionRequest(StrictModel):
    experiment_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")
    values: dict[str, str | float | int | None]
    explain: bool = True


class PredictionResponse(BaseModel):
    experiment_id: str
    dataset_key: str
    probability_positive: float
    threshold: float
    predicted_class: int
    predicted_label: str
    positive_label: str
    notes: list[str]
    explanation: dict[str, Any] | None
    explanation_error: str | None = None


class JobView(BaseModel):
    id: str
    status: str
    stage: str
    created: str
    updated: str
    config: dict[str, Any]
    round: int = 0
    completed_rounds: int = 0
    experiment_id: str | None = None
    cancel_requested: bool = False
    error: str | None = None
    clients: dict[str, Any] = Field(default_factory=dict)
    rounds: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)


class ExperimentPage(BaseModel):
    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int


class ComparisonRequest(StrictModel):
    ids: list[str] = Field(min_length=2, max_length=8)


class SettingsUpdate(StrictModel):
    defaults: TrainingRequest
