M2 shortcut audit — completed 25 October 2026

Scope
This package adds an exploratory popularity audit to the existing scheme, www, trailing-slash, length and separate-ablation evidence. It reuses five saved v3 test prediction arrays; no training, inference, network access or input changes were performed.

Key results
Original-source benign FPR: listed 4.54%, unlisted 11.91%.
Original-source malicious recall: listed 54.74%, unlisted 88.99%.
All 279 generated benign test URLs are listed, so a within-generated listed/unlisted comparison is unavailable.
Descriptive common-support standardization by source category, length band, path form and www status yields listed/unlisted benign FPR 6.43%/11.44% and malicious recall 79.13%/86.07%. This does not identify a causal popularity effect. Sparse cells, domain dependence and unmeasured differences remain.

Interpretation
Scheme encoding invariance is established. Remaining formatting sensitivity is a negative finding, not a successful general shortcut fix. Rank is an external descriptive proxy, not an original neural-model feature. The historical list was also used for augmentation. This is exploratory fixed-test analysis, not new independent validation or a deployment guarantee.

Contents
The results folder contains rank and joint strata, category comparisons, common-support tables, test membership and report.json with hashes, definitions and limitations. The script is provided for reproducibility. Full corpora, models and Tranco bytes are not redistributed here.

Optional reproduction — no rerun is needed
Dependencies: numpy, pandas, tldextract==5.3.1.
Place audit_m2_popularity_v1.py in the project folder and run:
python audit_m2_popularity_v1.py --root . --tranco PATH_TO_EXACT_RECOVERED_CSV --check-only
python audit_m2_popularity_v1.py --root . --tranco PATH_TO_EXACT_RECOVERED_CSV --run
Use the exact recovered historical CSV from the provenance evidence package, not a fresh daily download or the Git LFS pointer. The script validates its SHA256 and all frozen partitions and saved predictions. Model evaluation uses original y_prob.npy scores and checks CSV alignment within numerical tolerance with identical verdicts.

Manuscript changes
Section 4.1: method, raw rank associations, separated generated subset and standardized descriptive comparison.
Section 4.5: explicit limits on causal claims from single-seed ablations.
Reviewer 1 point 2: updated evidence and remaining-sensitivity framing.
All 18 manuscript tables, four figure media, citation anchors and bookmarks were retained and checked.
