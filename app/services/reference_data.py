"""Values transcribed from the supplied frontend design image (TA-EFL_Health_Interface_Showcase).

THESE ARE NOT EXPERIMENTAL RESULTS. They exist only so the interface can be
compared visually with the design. They are shown only when "Reference values"
mode is switched on, and every view that displays them carries the
``REFERENCE_BANNER`` text. Chart data that cannot be read from the image
(ROC curve points, per-round trust trajectories) is deliberately NOT included.
"""
from __future__ import annotations

REFERENCE_BANNER = ("REFERENCE VALUES transcribed from the design image - not produced by this "
                    "software and not experimental results.")

DASHBOARD = {
    "hospital_clients": 5,
    "training_rounds": 30,
    "total_records": 918,
    "test_roc_auc": 0.805,
    # Client Data Distribution (pie legend): name, records
    "distribution": [("Client 1", 120), ("Client 2", 115), ("Client 3", 118),
                     ("Client 4", 121), ("Client 5", 122)],
    "server_test_records": 322,
}

CLIENTS = [
    # Client ID, Dataset Size, Trust Score, Status, Last Round
    ("Client 1", 120, 0.85, "Active", 30),
    ("Client 2", 115, 0.80, "Active", 30),
    ("Client 3", 118, 0.78, "Active", 30),
    ("Client 4", 121, 0.40, "Malicious", 30),
    ("Client 5", 122, 0.82, "Active", 30),
]
CLIENT_TOTALS = {"client_records": 596, "server_test_records": 322, "total_records": 918}

TRAINING = {
    "dataset": "Heart Failure Prediction", "clients": 5, "rounds": 30, "learning_rate": 0.01,
    "batch_size": 32, "strategy": "Trust-Aware (TA-EFL)", "malicious_client": "Client 4",
    "trust_alpha": 0.70, "progress_round": 18, "progress_percent": 60, "global_auc": 0.803,
    "log": [
        "[Round 16] Client updates received.",
        "[Round 16] Trust scores updated.",
        "[Round 17] Aggregating client updates...",
        "[Round 17] Global model AUC = 0.801",
        "[Round 18] Client updates received.",
        "[Round 18] Aggregating client updates...",
        "[Round 18] Global model AUC = 0.803",
    ],
}

PREDICTION_INPUTS = {  # form inputs shown in the design (inputs, not results)
    "Age": 55, "Sex": "M", "ChestPainType": "TA", "RestingBP": 130, "Cholesterol": 250,
    "FastingBS": 1, "MaxHR": 150, "Oldpeak": 1.2,
}
PREDICTION_RESULT = {"headline": "High Risk", "probability": 0.82}

SHAP_VALUES = [("Age", 0.38), ("Cholesterol", 0.22), ("Max Heart Rate", -0.18),
               ("ST Depression", 0.16), ("Chest Pain Type", 0.12), ("Resting BP", 0.08),
               ("Fasting Blood Sugar", 0.06), ("Sex", -0.04)]
SHAP_STATEMENTS = [
    "Age increased the model's estimated risk.",
    "High cholesterol also contributed to higher risk.",
    "ST depression level increased the risk.",
    "Higher maximum heart rate reduced the risk.",
    "Sex had a small negative contribution.",
]

RESULTS_METRICS = [("ROC-AUC", 0.805), ("Accuracy", 0.742), ("Precision", 0.721),
                   ("Recall", 0.764), ("F1-Score", 0.742)]

TRUST_LATEST = [("Client 1", 0.85), ("Client 2", 0.80), ("Client 3", 0.78),
                ("Client 4", 0.40), ("Client 5", 0.82)]

REPORT_PREVIEW = {"dataset": "Heart Failure Prediction", "aggregation": "Trust-Aware (TA-EFL)",
                  "total_rounds": 30, "test_roc_auc": 0.805, "generated_on": "2025-05-12"}

TEAM = [("Manjunath K R", "1VA23CS093"), ("Mohit Khajuria", "1VA23CS097"),
        ("Sahil Kumar", "1VA23CS136"), ("Prem Kumar S", "1VA23CS120")]
