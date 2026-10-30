export type Config = {
  dataset_key: string;
  strategy: string;
  num_clients: number;
  num_rounds: number;
  local_epochs: number;
  learning_rate: number;
  batch_size: number;
  seed: number;
  attack_type: string;
  malicious_clients: number[];
  decision_threshold: number;
  partition: string;
  trust_alpha: number;
  trust_exclusion_threshold: number;
  label?: string;
  [key: string]: string | number | number[] | boolean | undefined;
};
export type Metrics = {
  roc_auc: number | null;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  n: number;
  threshold: number;
  prevalence: number;
  brier: number;
  ece: number;
  confusion_matrix: number[][];
  roc_curve: { fpr: number[]; tpr: number[] } | null;
  calibration_bins: {
    mean_predicted: number;
    observed_rate: number;
    n: number;
  }[];
};
export type Client = {
  cid: string;
  trust: number;
  weight: number | null;
  performance: number;
  similarity: number;
  num_examples: number;
  valid: boolean;
  excluded: boolean;
  reason: string;
  val_auc: number | null;
};
export type Round = {
  round: number;
  clients: Client[];
  model_updated: boolean;
  reference_auc: number | null;
  n_failures: number;
  trust_used_for_weighting: boolean;
};
export type Experiment = {
  id: string;
  created: string;
  status: string;
  config: Config;
  test_metrics: Metrics;
  current: boolean;
  origin: string;
  has_model: boolean;
  artifact_dir?: string;
  pipeline_version: number;
  dataset: { key: string; name: string; rows: number; file_sha256?: string };
  rounds_completed: number;
  rounds?: Round[];
  validation_history?: {
    round: number;
    auc: number;
    loss: number;
    accuracy: number;
  }[];
  trust_formula?: string;
  preprocessing?: { fitted_on: string };
  log?: string[];
  split_config?: Record<string, unknown>;
};
export type Feature = {
  name: string;
  label: string;
  kind: "numeric" | "binary" | "categorical";
  min_value: number | null;
  max_value: number | null;
  integer: boolean;
  unit: string;
  categories: string[];
  category_labels: string[];
  help: string;
};
export type Dataset = {
  key: string;
  name: string;
  description: string;
  source: string;
  source_url: string;
  licence: string;
  status: string;
  available: boolean;
  n_features: number;
  rows?: number;
  positives?: number;
  positive_rate?: number;
  duplicates?: number;
  missing?: Record<string, number>;
  warnings?: string[];
  notes: string[];
  sha256?: string;
  features: Feature[];
  positive_label: string;
  negative_label: string;
  target: string;
};
export type JobEvent = {
  type: string;
  round?: number;
  cid?: string;
  message?: string;
  sequence: number;
  received_at: string;
  auc?: number;
};
export type Job = {
  id: string;
  status: string;
  stage: string;
  created: string;
  updated: string;
  config: Config;
  round: number;
  completed_rounds: number;
  experiment_id: string | null;
  cancel_requested: boolean;
  error: string | null;
  clients: Record<string, JobEvent>;
  rounds: Round[];
  events: JobEvent[];
};
export type Session = {
  authenticated: boolean;
  registration_open: boolean;
  username: string | null;
  csrf: string | null;
};
export type Settings = {
  defaults: Config;
  strategies: Record<string, string>;
  attacks: Record<string, string>;
  python: string;
  pipeline_version: number;
  data_location: string;
  storage_location: string;
  execution: string;
  sessions: string;
  privacy: string;
};
export type DashboardData = {
  total_experiments: number;
  total_models: number;
  validated_datasets: number;
  latest: Experiment | null;
  recent: Experiment[];
  datasets: Dataset[];
  active_job: Job | null;
};
export type ExperimentPage = {
  items: Experiment[];
  total: number;
  page: number;
  page_size: number;
};
export type Prediction = {
  experiment_id: string;
  dataset_key: string;
  probability_positive: number;
  threshold: number;
  predicted_class: number;
  predicted_label: string;
  positive_label: string;
  notes: string[];
  explanation_error: string | null;
  explanation: null | {
    base_value: number;
    output_value: number;
    explained_label: string;
    output_scale: string;
    summary: string;
    statements: string[];
    caveats: string[];
    contributions: {
      feature: string;
      label: string;
      value: string;
      shap: number;
    }[];
  };
};
export type PersonalizationMetrics = {
  roc_auc: number | null;
  f1: number;
  precision: number;
  recall: number;
  brier: number;
  log_loss: number;
  ece: number;
  n: number;
};
export type PersonalizationHospitalSummary = {
  hospital_id: string;
  status: string;
  reason: string;
  n_calibration: number;
  n_test: number;
  prevalence: number;
};
export type PersonalizationRun = {
  id: string;
  created: string;
  status: string;
  method: string;
  label: string;
  dataset: { key: string; name: string };
  global_experiment: {
    id: string;
    strategy: string;
    attack_type: string;
    seed: number;
    decision_threshold: number;
  };
  data_protocol: {
    partition: string;
    dirichlet_alpha: number;
    num_hospitals: number;
    calibration_fraction: number;
  };
  calibration: {
    regularization: number;
    min_samples: number;
    min_per_class: number;
  };
  n_hospitals: number;
  n_personalized: number;
  matched_global: PersonalizationMetrics | null;
  matched_personalized: PersonalizationMetrics | null;
  hospitals: PersonalizationHospitalSummary[];
};
export type PersonalizationHospitalDetail = PersonalizationHospitalSummary & {
  calibration_positives: number;
  test_positives: number;
  calibrator: { a: number; b: number; n_samples: number } | null;
  global: PersonalizationMetrics & { auc_note?: string };
  personalized: (PersonalizationMetrics & { auc_note?: string }) | null;
};
export type PersonalizationRunDetail = Omit<PersonalizationRun, "hospitals"> & {
  formula: string;
  description: string;
  integrity: {
    global_model_unchanged: boolean;
    calibration_test_disjoint: boolean;
  };
  hospitals: PersonalizationHospitalDetail[];
};
export type PersonalizationReportMeta = {
  name: string;
  meta: {
    created: string;
    partition: string;
    alpha: number;
    seeds: number[];
    hospitals: number;
    attack: string;
    benchmark: string;
  };
};
type MeanSd = [number | null, number | null, number];
export type PersonalizationReport = {
  meta: PersonalizationReportMeta["meta"];
  coverage: {
    dataset: string;
    hospitals_personalised: number;
    hospitals_total: number;
    seeds: number[];
    seeds_with_personalised_hospital: number[];
  }[];
  groups: ({ dataset: string; group: string; label: string } & Record<
    "roc_auc" | "f1" | "precision" | "recall" | "brier" | "log_loss" | "ece",
    MeanSd
  >)[];
};
export type PersonalizedPrediction = {
  experiment_id: string;
  personalization_id: string;
  hospital_id: string;
  dataset_key: string;
  positive_label: string;
  threshold: number;
  global: {
    log_odds: number;
    probability_positive: number;
    predicted_label: string;
  };
  personalized: {
    log_odds: number;
    probability_positive: number;
    predicted_label: string;
  };
  calibration: {
    a: number;
    b: number;
    log_odds_adjustment: number;
    probability_change: number;
    n_calibration: number;
    regularization: number;
    method: string;
    hospital_local_test: {
      global: {
        brier: number;
        ece: number;
        roc_auc: number | null;
        f1: number;
      };
      personalized: {
        brier: number;
        ece: number;
        roc_auc: number | null;
        f1: number;
      };
    };
  };
  explanation: Prediction["explanation"];
  explanation_error: string | null;
  explanation_scope: string;
  notes: string[];
};
