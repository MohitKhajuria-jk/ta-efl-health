# TA-EFL Health – experiment results (pipeline v2)

Generated 2026-10-14 05:34 by `scripts/build_results.py` from `benchmark-20261014-052600.json` and `benchmark-20261014-053021.json` on Windows 11, Python 3.12.10, Flower 1.26.1.
Pipeline v2: preprocessing fitted from client training rows only. Results from the earlier pipeline are superseded and not used.

**Setup.** 5 simulated hospitals (iid), 30 rounds, seeds [1, 2, 3], configured attacker Client 5 (1 of 5); settings from config/settings.json. Mean ± SD over seeds. **Test** = held-out test split, never used for training, preprocessing, trust or thresholds. In-process Flower simulation, not a deployed network.

**Datasets.**

| Dataset | Records | Positive | Test n | Test prevalence | File SHA-256 |
|---|---|---|---|---|---|
| Heart Failure Prediction | 918 | 508 | 184 | 0.554 | 948420b084d8a3a0… |
| Pima Indians Diabetes | 768 | 268 | 154 | 0.351 | 27939f6c904b6c58… |
| Breast Cancer (Wisconsin Diagnostic) | 569 | 212 | 114 | 0.368 | bundled |
| Framingham CHD Risk | 4240 | 644 | 848 | 0.152 | 2c0e57dc0361b420… |
| Stroke Prediction | 5110 | 249 | 1022 | 0.049 | 644d473b05d27970… |

## 1. Strategy comparison under attack (test split)

| Dataset | Attack | Strategy | Test ROC-AUC | Test F1 | Test recall | Test ECE | Attacker final trust | Attacker last weight |
|---|---|---|---|---|---|---|---|---|
| Heart Failure Prediction | No attack | Trust-aware | 0.902 ± 0.026 | 0.847 ± 0.013 | 0.856 ± 0.015 | 0.079 ± 0.028 | — | — |
| Heart Failure Prediction | No attack | FedAvg | 0.902 ± 0.026 | 0.847 ± 0.013 | 0.856 ± 0.015 | 0.077 ± 0.027 | — | — |
| Heart Failure Prediction | No attack | Median | 0.902 ± 0.029 | 0.844 ± 0.019 | 0.853 ± 0.010 | 0.086 ± 0.031 | — | — |
| Heart Failure Prediction | Sign flipping | Trust-aware | 0.891 ± 0.028 | 0.828 ± 0.023 | 0.846 ± 0.023 | 0.118 ± 0.027 | 0.590 ± 0.008 | 0.134 ± 0.001 |
| Heart Failure Prediction | Sign flipping | FedAvg | 0.790 ± 0.024 | 0.762 ± 0.009 | 0.817 ± 0.006 | 0.096 ± 0.022 | 0.300 ± 0.110 | 0.200 ± 0.000 |
| Heart Failure Prediction | Sign flipping | Median | 0.900 ± 0.026 | 0.845 ± 0.024 | 0.859 ± 0.006 | 0.099 ± 0.028 | 0.595 ± 0.005 | — |
| Heart Failure Prediction | Update scaling | Trust-aware | 0.903 ± 0.025 | 0.852 ± 0.020 | 0.863 ± 0.026 | 0.060 ± 0.026 | 0.846 ± 0.034 | 0.191 ± 0.008 |
| Heart Failure Prediction | Update scaling | FedAvg | 0.903 ± 0.025 | 0.852 ± 0.020 | 0.863 ± 0.026 | 0.060 ± 0.026 | 0.845 ± 0.034 | 0.200 ± 0.000 |
| Heart Failure Prediction | Update scaling | Median | 0.902 ± 0.026 | 0.847 ± 0.022 | 0.856 ± 0.011 | 0.074 ± 0.029 | 0.883 ± 0.036 | — |
| Heart Failure Prediction | Gaussian noise | Trust-aware | 0.888 ± 0.044 | 0.822 ± 0.011 | 0.833 ± 0.043 | 0.084 ± 0.013 | 0.334 ± 0.050 | 0.059 ± 0.052 |
| Heart Failure Prediction | Gaussian noise | FedAvg | 0.607 ± 0.092 | 0.607 ± 0.075 | 0.592 ± 0.147 | 0.288 ± 0.088 | 0.321 ± 0.106 | 0.200 ± 0.000 |
| Heart Failure Prediction | Gaussian noise | Median | 0.902 ± 0.028 | 0.838 ± 0.022 | 0.853 ± 0.000 | 0.086 ± 0.023 | 0.271 ± 0.052 | — |
| Heart Failure Prediction | Label flipping | Trust-aware | 0.900 ± 0.025 | 0.837 ± 0.021 | 0.853 ± 0.010 | 0.132 ± 0.028 | 0.597 ± 0.003 | 0.135 ± 0.000 |
| Heart Failure Prediction | Label flipping | FedAvg | 0.896 ± 0.025 | 0.835 ± 0.022 | 0.850 ± 0.011 | 0.149 ± 0.035 | 0.592 ± 0.007 | 0.200 ± 0.000 |
| Heart Failure Prediction | Label flipping | Median | 0.902 ± 0.026 | 0.845 ± 0.021 | 0.853 ± 0.000 | 0.102 ± 0.028 | 0.598 ± 0.002 | — |
| Pima Indians Diabetes | No attack | Trust-aware | 0.814 ± 0.003 | 0.603 ± 0.011 | 0.562 ± 0.021 | 0.087 ± 0.017 | — | — |
| Pima Indians Diabetes | No attack | FedAvg | 0.814 ± 0.003 | 0.605 ± 0.014 | 0.562 ± 0.021 | 0.089 ± 0.017 | — | — |
| Pima Indians Diabetes | No attack | Median | 0.816 ± 0.004 | 0.598 ± 0.018 | 0.549 ± 0.011 | 0.079 ± 0.008 | — | — |
| Pima Indians Diabetes | Sign flipping | Trust-aware | 0.792 ± 0.010 | 0.570 ± 0.019 | 0.549 ± 0.028 | 0.108 ± 0.021 | 0.583 ± 0.016 | 0.134 ± 0.006 |
| Pima Indians Diabetes | Sign flipping | FedAvg | 0.732 ± 0.053 | 0.556 ± 0.074 | 0.568 ± 0.107 | 0.114 ± 0.026 | 0.419 ± 0.148 | 0.198 ± 0.000 |
| Pima Indians Diabetes | Sign flipping | Median | 0.807 ± 0.002 | 0.589 ± 0.020 | 0.543 ± 0.021 | 0.085 ± 0.011 | 0.599 ± 0.017 | — |
| Pima Indians Diabetes | Update scaling | Trust-aware | 0.819 ± 0.005 | 0.609 ± 0.025 | 0.549 ± 0.028 | 0.073 ± 0.009 | 0.783 ± 0.128 | 0.181 ± 0.034 |
| Pima Indians Diabetes | Update scaling | FedAvg | 0.819 ± 0.005 | 0.609 ± 0.022 | 0.543 ± 0.028 | 0.076 ± 0.008 | 0.783 ± 0.127 | 0.198 ± 0.000 |
| Pima Indians Diabetes | Update scaling | Median | 0.815 ± 0.005 | 0.601 ± 0.013 | 0.562 ± 0.011 | 0.080 ± 0.010 | 0.823 ± 0.139 | — |
| Pima Indians Diabetes | Gaussian noise | Trust-aware | 0.780 ± 0.015 | 0.584 ± 0.063 | 0.556 ± 0.096 | 0.104 ± 0.021 | 0.409 ± 0.144 | 0.101 ± 0.039 |
| Pima Indians Diabetes | Gaussian noise | FedAvg | 0.494 ± 0.183 | 0.391 ± 0.152 | 0.432 ± 0.185 | 0.263 ± 0.093 | 0.320 ± 0.254 | 0.198 ± 0.000 |
| Pima Indians Diabetes | Gaussian noise | Median | 0.811 ± 0.005 | 0.603 ± 0.015 | 0.562 ± 0.021 | 0.090 ± 0.013 | 0.340 ± 0.168 | — |
| Pima Indians Diabetes | Label flipping | Trust-aware | 0.804 ± 0.005 | 0.585 ± 0.020 | 0.549 ± 0.028 | 0.103 ± 0.010 | 0.594 ± 0.004 | 0.136 ± 0.005 |
| Pima Indians Diabetes | Label flipping | FedAvg | 0.800 ± 0.010 | 0.559 ± 0.029 | 0.519 ± 0.049 | 0.120 ± 0.018 | 0.590 ± 0.005 | 0.198 ± 0.000 |
| Pima Indians Diabetes | Label flipping | Median | 0.810 ± 0.006 | 0.602 ± 0.023 | 0.556 ± 0.019 | 0.083 ± 0.006 | 0.598 ± 0.001 | — |
| Breast Cancer (Wisconsin Diagnostic) | No attack | Trust-aware | 0.991 ± 0.005 | 0.944 ± 0.008 | 0.937 ± 0.036 | 0.096 ± 0.003 | — | — |
| Breast Cancer (Wisconsin Diagnostic) | No attack | FedAvg | 0.992 ± 0.005 | 0.944 ± 0.008 | 0.937 ± 0.036 | 0.096 ± 0.003 | — | — |
| Breast Cancer (Wisconsin Diagnostic) | No attack | Median | 0.991 ± 0.005 | 0.952 ± 0.012 | 0.952 ± 0.024 | 0.100 ± 0.009 | — | — |
| Breast Cancer (Wisconsin Diagnostic) | Sign flipping | Trust-aware | 0.976 ± 0.018 | 0.896 ± 0.066 | 0.921 ± 0.077 | 0.124 ± 0.026 | 0.590 ± 0.011 | 0.134 ± 0.003 |
| Breast Cancer (Wisconsin Diagnostic) | Sign flipping | FedAvg | 0.923 ± 0.042 | 0.794 ± 0.056 | 0.865 ± 0.090 | 0.137 ± 0.038 | 0.512 ± 0.046 | 0.197 ± 0.000 |
| Breast Cancer (Wisconsin Diagnostic) | Sign flipping | Median | 0.989 ± 0.006 | 0.921 ± 0.017 | 0.921 ± 0.036 | 0.090 ± 0.005 | 0.597 ± 0.001 | — |
| Breast Cancer (Wisconsin Diagnostic) | Update scaling | Trust-aware | 0.993 ± 0.005 | 0.960 ± 0.019 | 0.952 ± 0.024 | 0.086 ± 0.005 | 0.896 ± 0.029 | 0.196 ± 0.009 |
| Breast Cancer (Wisconsin Diagnostic) | Update scaling | FedAvg | 0.993 ± 0.005 | 0.960 ± 0.019 | 0.952 ± 0.024 | 0.086 ± 0.005 | 0.896 ± 0.028 | 0.197 ± 0.000 |
| Breast Cancer (Wisconsin Diagnostic) | Update scaling | Median | 0.991 ± 0.005 | 0.952 ± 0.012 | 0.952 ± 0.024 | 0.098 ± 0.008 | 0.892 ± 0.058 | — |
| Breast Cancer (Wisconsin Diagnostic) | Gaussian noise | Trust-aware | 0.969 ± 0.029 | 0.913 ± 0.046 | 0.913 ± 0.036 | 0.079 ± 0.008 | 0.398 ± 0.078 | 0.097 ± 0.021 |
| Breast Cancer (Wisconsin Diagnostic) | Gaussian noise | FedAvg | 0.857 ± 0.071 | 0.714 ± 0.067 | 0.770 ± 0.140 | 0.154 ± 0.056 | 0.392 ± 0.032 | 0.197 ± 0.000 |
| Breast Cancer (Wisconsin Diagnostic) | Gaussian noise | Median | 0.990 ± 0.005 | 0.941 ± 0.010 | 0.944 ± 0.027 | 0.096 ± 0.010 | 0.330 ± 0.081 | — |
| Breast Cancer (Wisconsin Diagnostic) | Label flipping | Trust-aware | 0.989 ± 0.008 | 0.928 ± 0.045 | 0.929 ± 0.063 | 0.174 ± 0.011 | 0.593 ± 0.006 | 0.131 ± 0.003 |
| Breast Cancer (Wisconsin Diagnostic) | Label flipping | FedAvg | 0.986 ± 0.009 | 0.916 ± 0.038 | 0.921 ± 0.060 | 0.212 ± 0.020 | 0.573 ± 0.016 | 0.197 ± 0.000 |
| Breast Cancer (Wisconsin Diagnostic) | Label flipping | Median | 0.990 ± 0.006 | 0.932 ± 0.007 | 0.929 ± 0.048 | 0.095 ± 0.001 | 0.599 ± 0.001 | — |
| Framingham CHD Risk | No attack | Trust-aware | 0.740 ± 0.010 | 0.125 ± 0.026 | 0.070 ± 0.016 | 0.040 ± 0.008 | — | — |
| Framingham CHD Risk | No attack | FedAvg | 0.740 ± 0.010 | 0.125 ± 0.026 | 0.070 ± 0.016 | 0.040 ± 0.008 | — | — |
| Framingham CHD Risk | No attack | Median | 0.738 ± 0.006 | 0.108 ± 0.029 | 0.059 ± 0.018 | 0.035 ± 0.008 | — | — |
| Framingham CHD Risk | Sign flipping | Trust-aware | 0.717 ± 0.014 | 0.134 ± 0.069 | 0.080 ± 0.047 | 0.054 ± 0.020 | 0.497 ± 0.063 | 0.125 ± 0.012 |
| Framingham CHD Risk | Sign flipping | FedAvg | 0.639 ± 0.038 | 0.286 ± 0.021 | 0.279 ± 0.074 | 0.182 ± 0.023 | 0.360 ± 0.200 | 0.200 ± 0.000 |
| Framingham CHD Risk | Sign flipping | Median | 0.725 ± 0.016 | 0.098 ± 0.009 | 0.054 ± 0.008 | 0.033 ± 0.012 | 0.537 ± 0.049 | — |
| Framingham CHD Risk | Update scaling | Trust-aware | 0.741 ± 0.010 | 0.143 ± 0.014 | 0.080 ± 0.009 | 0.028 ± 0.005 | 0.603 ± 0.017 | 0.177 ± 0.009 |
| Framingham CHD Risk | Update scaling | FedAvg | 0.740 ± 0.010 | 0.147 ± 0.006 | 0.083 ± 0.004 | 0.026 ± 0.005 | 0.607 ± 0.022 | 0.200 ± 0.000 |
| Framingham CHD Risk | Update scaling | Median | 0.739 ± 0.005 | 0.117 ± 0.034 | 0.065 ± 0.020 | 0.037 ± 0.006 | 0.777 ± 0.104 | — |
| Framingham CHD Risk | Gaussian noise | Trust-aware | 0.719 ± 0.007 | 0.265 ± 0.016 | 0.196 ± 0.031 | 0.057 ± 0.021 | 0.368 ± 0.039 | 0.100 ± 0.006 |
| Framingham CHD Risk | Gaussian noise | FedAvg | 0.652 ± 0.070 | 0.286 ± 0.007 | 0.292 ± 0.057 | 0.140 ± 0.065 | 0.408 ± 0.131 | 0.200 ± 0.000 |
| Framingham CHD Risk | Gaussian noise | Median | 0.734 ± 0.008 | 0.111 ± 0.033 | 0.062 ± 0.021 | 0.029 ± 0.010 | 0.410 ± 0.048 | — |
| Framingham CHD Risk | Label flipping | Trust-aware | 0.735 ± 0.010 | 0.103 ± 0.027 | 0.057 ± 0.016 | 0.086 ± 0.018 | 0.439 ± 0.085 | 0.107 ± 0.016 |
| Framingham CHD Risk | Label flipping | FedAvg | 0.724 ± 0.015 | 0.096 ± 0.057 | 0.054 ± 0.034 | 0.147 ± 0.007 | 0.261 ± 0.191 | 0.200 ± 0.000 |
| Framingham CHD Risk | Label flipping | Median | 0.727 ± 0.013 | 0.082 ± 0.007 | 0.044 ± 0.004 | 0.034 ± 0.014 | 0.511 ± 0.071 | — |
| Stroke Prediction | No attack | Trust-aware | 0.813 ± 0.038 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.014 ± 0.007 | — | — |
| Stroke Prediction | No attack | FedAvg | 0.813 ± 0.038 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.014 ± 0.007 | — | — |
| Stroke Prediction | No attack | Median | 0.815 ± 0.038 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.015 ± 0.005 | — | — |
| Stroke Prediction | Sign flipping | Trust-aware | 0.784 ± 0.042 | 0.013 ± 0.022 | 0.007 ± 0.012 | 0.018 ± 0.008 | 0.558 ± 0.023 | 0.133 ± 0.004 |
| Stroke Prediction | Sign flipping | FedAvg | 0.599 ± 0.141 | 0.036 ± 0.036 | 0.020 ± 0.020 | 0.068 ± 0.025 | 0.310 ± 0.275 | 0.200 ± 0.000 |
| Stroke Prediction | Sign flipping | Median | 0.808 ± 0.038 | 0.013 ± 0.022 | 0.007 ± 0.012 | 0.010 ± 0.003 | 0.588 ± 0.005 | — |
| Stroke Prediction | Update scaling | Trust-aware | 0.819 ± 0.036 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.013 ± 0.003 | 0.675 ± 0.019 | 0.166 ± 0.004 |
| Stroke Prediction | Update scaling | FedAvg | 0.819 ± 0.036 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.014 ± 0.004 | 0.661 ± 0.029 | 0.200 ± 0.000 |
| Stroke Prediction | Update scaling | Median | 0.815 ± 0.038 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.015 ± 0.006 | 0.784 ± 0.028 | — |
| Stroke Prediction | Gaussian noise | Trust-aware | 0.797 ± 0.046 | 0.011 ± 0.019 | 0.007 ± 0.012 | 0.015 ± 0.011 | 0.344 ± 0.084 | 0.063 ± 0.055 |
| Stroke Prediction | Gaussian noise | FedAvg | 0.605 ± 0.058 | 0.018 ± 0.032 | 0.013 ± 0.023 | 0.042 ± 0.016 | 0.275 ± 0.083 | 0.200 ± 0.000 |
| Stroke Prediction | Gaussian noise | Median | 0.810 ± 0.040 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.011 ± 0.004 | 0.286 ± 0.072 | — |
| Stroke Prediction | Label flipping | Trust-aware | 0.797 ± 0.041 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.101 ± 0.016 | 0.520 ± 0.102 | 0.115 ± 0.020 |
| Stroke Prediction | Label flipping | FedAvg | 0.768 ± 0.012 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.172 ± 0.004 | 0.345 ± 0.291 | 0.200 ± 0.000 |
| Stroke Prediction | Label flipping | Median | 0.815 ± 0.036 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.015 ± 0.004 | 0.595 ± 0.008 | — |

Trust for FedAvg and Median is monitored only (not used for weighting); Median has no per-client weights (—).

## 2. Trust performance signal: zero-weight decisions

Client-rounds given zero weight by the trust-aware strategy (30 rounds × 5 clients × 3 seeds per row).

| Dataset | Attack | Honest zeroed (absolute) | Honest zeroed (relative) | Attacker zeroed (absolute) | Attacker zeroed (relative) |
|---|---|---|---|---|---|
| Heart Failure Prediction | No attack | 0/450 (0.0%) | 0/450 (0.0%) | — | — |
| Heart Failure Prediction | Sign flipping | 0/360 (0.0%) | 0/360 (0.0%) | 15/90 (16.7%) | 13/90 (14.4%) |
| Heart Failure Prediction | Update scaling | 0/360 (0.0%) | 0/360 (0.0%) | 0/90 (0.0%) | 0/90 (0.0%) |
| Heart Failure Prediction | Gaussian noise | 0/360 (0.0%) | 0/360 (0.0%) | 67/90 (74.4%) | 44/90 (48.9%) |
| Heart Failure Prediction | Label flipping | 0/360 (0.0%) | 0/360 (0.0%) | 5/90 (5.6%) | 3/90 (3.3%) |
| Pima Indians Diabetes | No attack | 0/450 (0.0%) | 0/450 (0.0%) | — | — |
| Pima Indians Diabetes | Sign flipping | 0/360 (0.0%) | 0/360 (0.0%) | 17/90 (18.9%) | 10/90 (11.1%) |
| Pima Indians Diabetes | Update scaling | 0/360 (0.0%) | 0/360 (0.0%) | 0/90 (0.0%) | 0/90 (0.0%) |
| Pima Indians Diabetes | Gaussian noise | 0/360 (0.0%) | 0/360 (0.0%) | 58/90 (64.4%) | 30/90 (33.3%) |
| Pima Indians Diabetes | Label flipping | 0/360 (0.0%) | 0/360 (0.0%) | 5/90 (5.6%) | 2/90 (2.2%) |
| Breast Cancer (Wisconsin Diagnostic) | No attack | 0/450 (0.0%) | 0/450 (0.0%) | — | — |
| Breast Cancer (Wisconsin Diagnostic) | Sign flipping | 0/360 (0.0%) | 0/360 (0.0%) | 9/90 (10.0%) | 9/90 (10.0%) |
| Breast Cancer (Wisconsin Diagnostic) | Update scaling | 0/360 (0.0%) | 0/360 (0.0%) | 0/90 (0.0%) | 0/90 (0.0%) |
| Breast Cancer (Wisconsin Diagnostic) | Gaussian noise | 0/360 (0.0%) | 0/360 (0.0%) | 30/90 (33.3%) | 30/90 (33.3%) |
| Breast Cancer (Wisconsin Diagnostic) | Label flipping | 0/360 (0.0%) | 0/360 (0.0%) | 3/90 (3.3%) | 3/90 (3.3%) |
| Framingham CHD Risk | No attack | 6/450 (1.3%) | 0/450 (0.0%) | — | — |
| Framingham CHD Risk | Sign flipping | 10/360 (2.8%) | 0/360 (0.0%) | 81/90 (90.0%) | 11/90 (12.2%) |
| Framingham CHD Risk | Update scaling | 12/360 (3.3%) | 0/360 (0.0%) | 6/90 (6.7%) | 0/90 (0.0%) |
| Framingham CHD Risk | Gaussian noise | 0/360 (0.0%) | 0/360 (0.0%) | 82/90 (91.1%) | 48/90 (53.3%) |
| Framingham CHD Risk | Label flipping | 16/360 (4.4%) | 0/360 (0.0%) | 87/90 (96.7%) | 15/90 (16.7%) |
| Stroke Prediction | No attack | 0/450 (0.0%) | 0/450 (0.0%) | — | — |
| Stroke Prediction | Sign flipping | 0/360 (0.0%) | 0/360 (0.0%) | 27/90 (30.0%) | 16/90 (17.8%) |
| Stroke Prediction | Update scaling | 0/360 (0.0%) | 0/360 (0.0%) | 0/90 (0.0%) | 0/90 (0.0%) |
| Stroke Prediction | Gaussian noise | 0/360 (0.0%) | 0/360 (0.0%) | 71/90 (78.9%) | 35/90 (38.9%) |
| Stroke Prediction | Label flipping | 0/360 (0.0%) | 0/360 (0.0%) | 13/90 (14.4%) | 8/90 (8.9%) |

## 3. Findings (computed from the tables above)

* **Sign flipping:** trust-aware test ROC-AUC was higher than FedAvg on 5 of 5 datasets (Heart Failure Prediction 0.891 vs 0.790; Pima Indians Diabetes 0.792 vs 0.732; Breast Cancer (Wisconsin Diagnostic) 0.976 vs 0.923; Framingham CHD Risk 0.717 vs 0.639; Stroke Prediction 0.784 vs 0.599). Median was equal or better than trust-aware on Heart Failure Prediction, Pima Indians Diabetes, Breast Cancer (Wisconsin Diagnostic), Framingham CHD Risk, Stroke Prediction.
* **Gaussian noise:** trust-aware test ROC-AUC was higher than FedAvg on 5 of 5 datasets (Heart Failure Prediction 0.888 vs 0.607; Pima Indians Diabetes 0.780 vs 0.494; Breast Cancer (Wisconsin Diagnostic) 0.969 vs 0.857; Framingham CHD Risk 0.719 vs 0.652; Stroke Prediction 0.797 vs 0.605). Median was equal or better than trust-aware on Heart Failure Prediction, Pima Indians Diabetes, Breast Cancer (Wisconsin Diagnostic), Framingham CHD Risk, Stroke Prediction.
* **Update scaling:** the attacker was given zero weight in 0/450 (0.0%) of its rounds; its update points in the honest direction, so this policy does not detect it.
* **Label flipping** by one client reduced FedAvg test ROC-AUC by 0.005–0.045 across datasets.
* **Honest hospitals** were wrongly given zero weight in 0/9450 (0.0%) of client-rounds with the relative signal, versus 44/9450 (0.5%) with the original absolute signal.
* **Calibration:** expected calibration error ranged from 0.007 to 0.375 across runs, so outputs are model scores, not calibrated risks.
* **Imbalanced datasets** (Framingham CHD Risk, Stroke Prediction): F1/recall at threshold 0.5 are low by construction; report ROC-AUC and consider class weighting 'balanced'.

## 4. Scope

Simulated hospitals in one program, one configured attacker per run. Low trust is a policy output, not proof of malicious behaviour, and says nothing about real-world attack detection. Research prototype; not a medical device.
