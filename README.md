# PhishGuard Ghana: Research Revision

Research code and saved experiments for malicious URL screening using a
CNN-LSTM, comparative models, rule annotations and tamper-evident logging.

## Scope

The positive class combines phishing, malware and defacement URLs.
Ghana motivates the application setting; these experiments do not establish
performance on Ghanaian or mobile-money traffic.

This branch contains revised research artifacts alongside historical files.
Use the versioned v3 folders for the original revised experiments. Legacy
root-level models, datasets and metrics belong to earlier experiments and
should not be mixed with v3 results.

## Main artifacts

- dataset_fix_outputs_v3/: frozen training, validation and test partitions,
  containing 35,664, 7,668 and 7,653 rows respectively, plus numeric features.
- corrected_results_v3/: CNN-LSTM models, predictions and metrics for
  seeds 42, 43, 44, 45 and 46.
- baseline_results_v3/: baseline tuning, selected configurations and
  five-seed evaluation results.
- transformer_baseline_v3/: compact Transformer experiments.
- published_baseline_results/: adapted DomURLs-BERT evaluation outputs
  and training protocols. Its weights, tokenizer assets, notebook and
  selected integration code are excluded pending redistribution review.
- published_cpu_comparison_v3/: saved CPU comparison outputs.
- coverage_development_v1/, coverage_pilot_v1/ and coverage_frozen_v1/:
  broader-data development inputs, training runs and frozen protocol.
- reserved_mendeley_holdout/ and reserved_holdout_results_v1/:
  separate-corpus holdout inputs and evaluation results.
- category_evaluation_v3/, rule_evaluation_v3/,
  attribution_evaluation_v3/ and secure_logging_evaluation_v3/:
  category, rule, attribution and logging analyses.
- shortcut_checks_v3/, m2_popularity_audit/, adaptive_evasion_v3/,
  controlled_robustness_v1/ and workflow_preservation_audit/:
  shortcut checks and synthetic or controlled-workflow stress tests.
- controlled_experiments_v3/, masked_bilstm_control_v1/ and
  cpu_benchmarks_v3/: sensitivity, recurrent masking and timing experiments.
- release_support/: supporting analyses and provenance records.
  Its manuscript reference is historical, not necessarily the submission version.

## Interpretation

PhiUSIIL became development data after its initial evaluation and must not
be presented as untouched independent test evidence for later models.
Controlled workflow tests use local dummy simulations; they do not establish
preservation of live phishing functionality.

Intervals across training seeds describe variation on fixed data, not
population generalization. CPU measurements apply to the recorded hardware
and implementations. These artifacts do not establish production readiness,
legal admissibility or resistance to every log-rewriting attacker.

## Downloading model artifacts

This repository uses Git LFS. A complete checkout requires Git LFS and
access to the corresponding LFS objects:

    git lfs install
    git lfs pull

A small LFS pointer file is not a usable model checkpoint.

## Reproduction and provenance

Consult each experiment's saved protocol, report and metrics before running
its script. Historical scripts may require additional inputs or dependencies.
Included file hashes support integrity checks; copying or packaging files
does not constitute fresh inference or full retraining reproduction.

The excluded DomURLs-BERT components mean this checkout alone cannot
reproduce every published-baseline training or inference step.

## Data sources and rights

See THIRD_PARTY_NOTICES.md for source attribution, source-specific terms
and unresolved redistribution questions. Inclusion does not assign a blanket
license to third-party datasets, code or pretrained components.

URL datasets may contain malicious addresses. Treat them as research data;
do not open their links during routine inspection.
