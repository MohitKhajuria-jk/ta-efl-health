const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ImageRun, BorderStyle,
} = require("docx");

const p = (text, opts = {}) => new Paragraph({
  children: [new TextRun({ text, ...opts.run })],
  spacing: { after: 160, line: 300 },
  alignment: opts.align || AlignmentType.JUSTIFIED,
  ...opts.para,
});
const h1 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 160 } });
const h2 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 120 } });
const eq = t => new Paragraph({
  children: [new TextRun({ text: t, italics: true })],
  alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
});
const caption = t => new Paragraph({
  children: [new TextRun({ text: t, size: 18, italics: true })],
  alignment: AlignmentType.CENTER, spacing: { after: 240 },
});
const img = (path, w, hgt) => new Paragraph({
  children: [new ImageRun({ data: fs.readFileSync(path), type: "png", transformation: { width: w, height: hgt } })],
  alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 },
});

const cell = (t, bold = false) => new TableCell({
  width: { size: 2340, type: WidthType.DXA },
  children: [new Paragraph({ children: [new TextRun({ text: t, bold, size: 20 })], alignment: AlignmentType.CENTER })],
});
const row = (vals, bold = false) => new TableRow({ children: vals.map(v => cell(v, bold)) });

const e1Table = new Table({
  columnWidths: [2340, 2340, 2340, 2340],
  width: { size: 9360, type: WidthType.DXA },
  rows: [
    row(["Malicious clients", "TA-EFL (ours)", "FedAvg", "Coord. median"], true),
    row(["0 / 6", "0.795 ± 0.009", "0.796 ± 0.010", "0.796 ± 0.010"]),
    row(["1 / 6", "0.793 ± 0.009", "0.794 ± 0.008", "0.788 ± 0.003"]),
    row(["2 / 6", "0.796 ± 0.008", "0.765 ± 0.022", "0.789 ± 0.008"]),
    row(["3 / 6", "0.793 ± 0.008", "0.733 ± 0.050", "0.789 ± 0.008"]),
  ],
});

const e2Table = new Table({
  columnWidths: [3120, 3120, 3120],
  width: { size: 9360, type: WidthType.DXA },
  rows: [
    row(["Trust signal", "Final AUC", "Detection margin"], true),
    row(["Update similarity only (α = 1)", "0.786", "−0.055"]),
    row(["Explanation consistency only (α = 0)", "0.798", "0.408"]),
    row(["Both (α = 0.5)", "0.796", "0.155"]),
  ],
});

const doc = new Document({
  styles: { default: { document: { run: { font: "Times New Roman", size: 22 } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Trust-Aware Explainable Federated Learning for Personalized Healthcare Risk Prediction", bold: true, size: 32 })],
        alignment: AlignmentType.CENTER, spacing: { after: 200 },
      }),
      p("[Author Name], [Institution], [email]", { align: AlignmentType.CENTER, run: { size: 20 } }),

      h1("Abstract"),
      p("Federated learning (FL) enables hospitals to collaboratively train clinical risk-prediction models without sharing patient records, but it is vulnerable to malicious or low-quality participants, and its predictions are typically opaque to clinicians. Robustness and explainability have so far been addressed as separate problems. We propose TA-EFL, a trust-aware explainable federated learning framework that couples them: each participant's trustworthiness is scored not only from the geometry of its model updates but also from the consistency of its model explanations. Concretely, the server computes a signed SHAP attribution signature of an extrapolated model θg + κ(θk − θg) on a small public reference set, exposing where each client is pushing the global model; because the signature is derived server-side from the submitted weights, it cannot be spoofed. Trust scores evolve via an exponential moving average and drive a trust-weighted FedAvg aggregation with hard exclusion of untrusted clients. On a heart-disease risk-prediction task partitioned non-IID across six simulated hospitals, TA-EFL sustains a test AUC of 0.793 with half of the clients malicious (label flipping, Gaussian noise, and scaled model poisoning), whereas standard FedAvg degrades to 0.733; unlike coordinate-median aggregation, our method additionally identifies which participants are malicious. An ablation shows the explanation-consistency signal is essential: update-space similarity alone fails to separate attackers from honest clients (detection margin −0.055), while explanation consistency achieves a margin of 0.408. Every prediction is delivered with an exact per-patient SHAP explanation, making the system auditable end to end."),
      p("Keywords — federated learning, explainable AI, SHAP, trust, Byzantine robustness, healthcare risk prediction", { run: { italics: true } }),

      h1("1. Introduction"),
      p("Clinical risk-prediction models improve with more diverse training data, yet privacy regulation and institutional policy prevent hospitals from pooling patient records. Federated learning resolves this tension by exchanging model parameters instead of data: each hospital trains locally and a central server aggregates the resulting updates into a global model. Two obstacles, however, limit the deployment of FL in healthcare. First, the aggregation step is fragile: a single participant submitting corrupted updates, whether through faulty data pipelines, mislabeled records, or deliberate poisoning, can degrade or hijack the global model. Second, clinical adoption demands interpretability: a clinician must be able to see why a patient was flagged as high risk before acting on the prediction."),
      p("These two requirements are usually pursued independently, with Byzantine-robust aggregation rules on one side and post-hoc explanation methods on the other. The central observation of this work is that they are mutually reinforcing: the explanation behavior of a participant's model is itself a powerful signal of that participant's trustworthiness. A model trained on poisoned labels may achieve a plausible loss, yet it attributes risk to clinical features in a systematically inverted or distorted way. We operationalize this observation in TA-EFL, which makes three contributions:"),
      p("(1) A dual-signal trust engine that combines update-space anomaly detection (cosine similarity to the cohort's coordinate-wise median update, with a magnitude penalty) with a novel explanation-space signal based on SHAP attribution signatures computed on a shared public reference set."),
      p("(2) An extrapolated, server-side signature construction. We show that naive per-round signatures fail against label-flipping attacks: because each client trains only a few epochs from the global model, its submitted weights barely drift from the honest global model, and absolute-value SHAP profiles are additionally invariant to sign inversion. Computing a signed signature of the extrapolated model θg + κ(θk − θg) amplifies the direction in which the client is pushing the model and restores detectability; because the server computes it from the submitted weights, no self-reported quantity is trusted."),
      p("(3) An end-to-end auditable system, implemented as a desktop application, in which trust scores, aggregation weights, and exact per-patient SHAP explanations are all inspectable by an operator, and evaluated under label-flipping, Gaussian-noise, and scaled model-poisoning attacks at up to 50% malicious participation."),

      h1("2. Related Work"),
      p("Federated learning in healthcare. FedAvg [McMahan et al., 2017] established parameter averaging weighted by local sample counts, and has been applied to clinical prediction across institutions, with FedProx [Li et al., 2020] addressing statistical heterogeneity. These methods presume honest participants."),
      p("Byzantine-robust aggregation. Krum [Blanchard et al., 2017], coordinate-wise median and trimmed mean [Yin et al., 2018] tolerate a bounded fraction of arbitrary updates by geometric filtering in parameter space. They provide robustness but no attribution: the operator learns that aggregation was protected, not which participants misbehaved, and they discard information from honest heterogeneous clients. Reputation- and trust-based FL schemes weight clients by historical performance or validation accuracy, but rely on signals a strategic adversary can imitate."),
      p("Explainability in FL. SHAP [Lundberg and Lee, 2017] provides additively decomposable feature attributions; for linear models the values are exact and computable in closed form. Prior work has used explanations to interpret federated models after training. To our knowledge, using explanation consistency as an online trust signal during federated aggregation, and in particular on extrapolated models to overcome small per-round drift, has not been explored."),

      h1("3. Methodology"),
      h2("3.1 System model"),
      p("K hospitals hold private datasets Dk = {(xi, yi)} of tabular patient features with a binary outcome. Training proceeds in rounds t = 1…T. The server broadcasts global parameters θg(t); each client k trains locally for E epochs of mini-batch gradient descent on a regularized logistic-regression objective and submits θk(t). A small public reference set Xref (synthetic or public data, never private records) is fixed in advance and known to the server. An unknown subset of clients is malicious, employing (a) label flipping, training on inverted labels; (b) additive Gaussian noise on the submitted parameters; or (c) scaled model poisoning, submitting θg + λ(θk − θg) with λ ≫ 1."),
      h2("3.2 Explanation signatures"),
      p("For a linear risk model with weights w, the exact SHAP value of feature j for patient x in log-odds space is φj(x) = wj (xj − E[xj]). We define a client's attribution signature as the signed, L2-normalized vector of global feature attributions on the reference set, sk ∝ w ⊙ σ(Xref), where σ(Xref) is the per-feature standard deviation. The sign is essential: a label-flipped model has w ≈ −w* and an absolute-value profile identical to the honest one, but a signed signature that is anti-correlated with the cohort."),
      p("A second failure mode is more subtle. Since clients train only E epochs from θg each round, a poisoned client's submitted weights remain close to the (honest) global weights; the poison resides in the update direction rather than in the resulting model, and the signature of θk is nearly indistinguishable from honest signatures after the first round. We therefore compute the signature of the extrapolated model"),
      eq("θ̃k = θg + κ (θk − θg),   κ > 1,"),
      p("which amplifies the client's pushing direction until its attribution consequences become visible. Crucially, θ̃k is computed by the server from the submitted θk, so the signal cannot be falsified by the client, in contrast to self-reported explanations or validation metrics."),
      h2("3.3 Trust engine"),
      p("Let uk = θk − θg denote client k's update, m the coordinate-wise median of all updates, and s̄ the coordinate-wise median signature. We define"),
      eq("s_upd(k) = max(0, cos(uk, m)) · min(1, ||m||med / ||uk||),"),
      eq("s_exp(k) = max(0, cos(sk, s̄)),"),
      p("where the second factor of s_upd penalizes abnormal update magnitudes (defeating scaling attacks that preserve direction). Trust evolves by exponential moving average:"),
      eq("Tk(t) = β Tk(t−1) + (1 − β) [ α s_upd(k) + (1 − α) s_exp(k) ]."),
      h2("3.4 Trust-weighted aggregation"),
      p("Aggregation weights combine data volume with trust: pk ∝ nk · Tk^γ, with any client whose trust falls below a threshold τ excluded outright (weight zero). The exponent γ sharpens the penalty on partially trusted clients; τ provides a hard guarantee that persistently anomalous participants cannot influence the global model. The global update is θg ← Σk pk θk. When trust scoring is disabled the system reduces exactly to FedAvg, which we use as a baseline alongside coordinate-wise median aggregation."),
      h2("3.5 Personalization and clinician-facing explanations"),
      p("After federation, each hospital may fine-tune the global model on local data for a small number of epochs, yielding a personalized model that reflects its population while retaining federated knowledge. At inference, every prediction is accompanied by the exact SHAP decomposition of its log-odds, presented as a signed per-feature contribution chart, so that a clinician can audit which factors (e.g., ST depression, exercise-induced angina, maximum heart rate) drove an individual risk estimate."),

      h1("4. Experimental Setup"),
      p("Task and data. Binary heart-disease risk prediction with 13 standard clinical features (age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, ST slope, number of major vessels, thalassemia). Experiments reported here use a synthetic dataset (n = 1500, 51.6% positive) generated from a clinically plausible risk function calibrated to the UCI Heart Disease distribution; the framework loads the real UCI/Kaggle CSV identically when available. Data is split 75/25 into a federated training pool and a held-out test set, and the training pool is partitioned across K = 6 clients by a Dirichlet distribution over labels (α = 0.5 unless stated), simulating heterogeneous hospital populations."),
      p("Protocol. T = 30 rounds; E = 3 local epochs; learning rate 0.15; batch size 32; L2 regularization 10−3. Trust hyperparameters: α = 0.5, β = 0.6, γ = 2, τ = 0.25, κ = 4. All results are means ± standard deviation over 3 seeds. The reference set contains 200 rows. Metrics: test AUC of the global model, and the detection margin, defined as (minimum honest-client trust) − (maximum attacker trust), which is positive if and only if a threshold exists that separates all attackers from all honest clients."),

      h1("5. Results"),
      h2("5.1 Robustness to increasing attack fraction"),
      p("Table 1 and Figure 1 compare final test AUC as the malicious fraction grows from 0/6 to 3/6 (adding, in order: a label flipper, a Gaussian-noise client, and a scaled poisoner). TA-EFL is statistically indistinguishable from the baselines under no attack, and maintains AUC ≈ 0.793–0.796 at every attack level, while FedAvg degrades progressively to 0.733 ± 0.050 at 3/6 malicious. Coordinate-wise median also remains robust, as expected of a geometric filter; the advantage of TA-EFL over median is attributional: it produces per-client trust trajectories that identify which participants are malicious (all three attackers receive the three lowest trust scores in every run), enabling exclusion, audit, and operator intervention that median aggregation cannot support."),
      e1Table,
      caption("Table 1. Final test AUC (mean ± sd over 3 seeds) versus number of malicious clients."),
      img("results/study_e1_robustness.png", 480, 309),
      caption("Figure 1. Robustness to increasing attack fraction."),
      h2("5.2 The explanation signal is essential"),
      p("Table 2 ablates the two trust signals at 2/6 malicious. Update-space similarity alone yields a negative detection margin (−0.055): near convergence, honest clients' updates are small and noisy, and cosine similarity to the median update no longer separates attackers reliably. Explanation consistency alone achieves the largest margin (0.408) while preserving accuracy, and the combined signal remains comfortably positive (0.155). This is the central empirical finding of this work: the explanation space carries discriminative information about participant trustworthiness that the update space does not."),
      e2Table,
      caption("Table 2. Trust-signal ablation at 2/6 malicious clients."),
      img("results/study_e2_ablation.png", 480, 274),
      caption("Figure 2. Ablation of trust signals: AUC and detection margin."),
      h2("5.3 Extrapolation factor"),
      p("Figure 3 sweeps κ against a label-flipping client. Without extrapolation (κ = 1) the margin is negative (−0.105), confirming that per-round drift is too small for direct signature comparison. Detection peaks at κ = 4 (margin 0.164) and deteriorates for very large κ, where extrapolation amplifies benign heterogeneity in honest clients' updates as strongly as the attack, producing an inverted-U response that motivates our default κ = 4."),
      img("results/study_e3_kappa.png", 440, 271),
      caption("Figure 3. Effect of signature extrapolation factor κ on label-flip detection."),
      h2("5.4 No false exclusions under heterogeneity"),
      p("A trust mechanism is only deployable if statistical heterogeneity is not mistaken for malice. With all clients honest, we sweep the Dirichlet concentration from α = 0.1 (extreme non-IID) to α = 5. The minimum honest-client trust remains between 0.383 and 0.430 in all settings, well above the exclusion threshold τ = 0.25 (Figure 4): no honest hospital is ever excluded, even under severe population shift."),
      img("results/study_e4_noniid.png", 440, 271),
      caption("Figure 4. Minimum honest-client trust under increasing heterogeneity (no attack)."),

      h1("6. Discussion and Limitations"),
      p("Privacy of the reference set and signatures. Signatures are computed on public or synthetic reference data and reveal only a normalized global attribution direction; no patient-level information leaves any hospital beyond model parameters, as in standard FL. Differentially private noise could be added to submitted parameters at a quantifiable cost to detection margin, which we leave to future work."),
      p("Adaptive adversaries. Our extrapolated signature is unspoofable in the sense that it derives from the submitted weights, but a sophisticated adversary aware of the mechanism could constrain its poisoned updates to remain attribution-consistent, trading attack strength for stealth. Characterizing this trade-off, and randomizing κ or the reference set across rounds to raise the cost of adaptation, are natural extensions."),
      p("Model class. Closed-form SHAP makes linear models an ideal testbed, and linear risk scores remain clinically prevalent; extending signatures to gradient-boosted trees or neural networks via TreeSHAP or sampling-based SHAP is mechanically straightforward but increases per-round cost, and the extrapolation θg + κ(θk − θg) remains well defined for any parameter vector."),
      p("Scale. Experiments use six simulated clients, matching the cross-silo hospital setting; evaluating hundreds of cross-device clients and real multi-site clinical datasets (e.g., MIMIC-IV) is ongoing work."),

      h1("7. Conclusion"),
      p("TA-EFL couples robustness and explainability in federated healthcare risk prediction, using the explanation behavior of participants' models, evaluated on extrapolated parameters and computed entirely server-side, as a trust signal that update-space geometry alone cannot provide. The system sustains predictive performance with up to half of the participants malicious, identifies the malicious participants explicitly, never falsely excludes honest hospitals under severe heterogeneity, and delivers an exact per-patient explanation with every prediction."),

      h1("References"),
      p("[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017."),
      p("[2] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith. Federated Optimization in Heterogeneous Networks. MLSys 2020."),
      p("[3] P. Blanchard, E. M. El Mhamdi, R. Guerraoui, J. Stainer. Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. NeurIPS 2017."),
      p("[4] D. Yin, Y. Chen, K. Ramchandran, P. Bartlett. Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. ICML 2018."),
      p("[5] S. Lundberg, S.-I. Lee. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017."),
      p("[6] D. Dua, C. Graff. UCI Machine Learning Repository: Heart Disease Data Set. University of California, Irvine."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("TA-EFL_paper_draft.docx", buf);
  console.log("written");
});
