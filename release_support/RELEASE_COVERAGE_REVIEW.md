# Release coverage review

The first candidate passed its local archive consistency check, but the manuscript-to-file audit found omitted later studies. Candidate 1 contains 846 listed payload files, totalling 4,750,499,408 bytes. Its recorded outer SHA256 is 73c97f9eeab5081275c1111058fc5669156c75ee8ddc94a32e88d5bab1a43fda. The outer ZIP was not supplied for independent rehashing. Its manifest was reviewed for inventory and structure, not proof that every study reproduced.

## Corrected inclusion

prepare_release_v2.py retains the original core checks and adds the actual output directories attribution_evaluation_v3 and secure_logging_evaluation_v3, actual broader-data runs in coverage_pilot_v1, prepared inputs in coverage_development_v1 and the frozen reserved_mendeley_holdout directory. It also includes .sha256 sidecars and root-level Colab notebooks. The frozen coverage protocol alone is not the five broader-data models.

Extract this support ZIP into the project ROOT, preserving its two top-level directories: m2_popularity_audit and release_support. Keep the old candidate unchanged. Run v2 --check-only before creating a new candidate. No training is required.

## Manuscript to artifact map

| Manuscript evidence | Candidate 1 inventory | Required resolution |
| --- | --- | --- |
| Tables 2-4 original data, features and architecture | Frozen partitions, feature files and five main checkpoints present | Retain existing hash checks and original scripts |
| Table 5 and Figure 1, low-FPR operating points | Five main run predictions present; secondary aggregation outputs omitted | Archived analysis results and code supplied under release_support/analyses/analysis_v3 |
| Table 6 original baselines and paired comparisons | Forty baseline runs present; paired comparison aggregation omitted | Supplied under release_support/analyses/baseline_analysis_v3; original testing family unchanged |
| Table 6 masked control | Five masked models and test predictions present | Preserve control protocol and completion hashes |
| Table 7 sensitivity | Eight experiment folders and reference main run present | Retain single-seed exploratory scope |
| Table 8 fixture stress and working replay | Fixture rounds and replay records present | Retain original disabled-page versus later dummy-workflow distinction |
| Table 9 stage timing | Timing records present | Retain hardware and noise interpretation |
| Figure 4 and Table 13 plain tamper suite | Tamper files present inside original seed folders | Retain exact record schema and reference assumptions |
| Tables 13-14 keyed logging | Script present, output directory omitted | Include secure_logging_evaluation_v3 with v2 |
| Table 10 broader-data holdout | Holdout scores and frozen manifest present; actual broader-data models and prepared inputs omitted | Include coverage_pilot_v1, coverage_development_v1 and reserved_mendeley_holdout with v2 |
| Tables 11-12 rules | Rule metrics and worked examples present | Retain source-label and parsing limits |
| Tables 15-16 and Figure 3 attribution | Script present, full outputs omitted | Include attribution_evaluation_v3 with v2 |
| Tables 17-18 compact/published Transformer | Compact models and five seed-labelled published checkpoints present | Add the actual Colab notebook and pinned tokenizer/config files; verify checkpoint-to-evaluation identity |
| Table 19 DSR | Interpretation of existing experiments | Current manuscript reference supplied; no practitioner validation claimed |
| Dataset provenance and generation | Only the old failed collection report was included | Curated final provenance reports and audit code supplied here; original full source inputs remain separately fingerprinted |
| Popularity and length/format audit | M2 popularity absent; raw shortcut records present | Completed M2 audit and shortcut aggregation supplied here |

## Archived analysis code

The analysis scripts are preserved unchanged. They expect source archives under an upload directory relative to their parent analysis folder; some also expect original augmentation and snapshots. They do not become directly runnable merely by extracting their CSV results. Use a separate working copy and supply their documented source archives when replaying them. The core release builder independently recalculates core metrics from expanded CSVs without loading model weights. Full fresh inference, retraining and execution of every later analysis have not been independently repeated by this coverage review.

## Still required before publication

Export the actual DomURLs_BERT Colab notebook used for training/evaluation and retain its frozen protocol. Include config.json, tokenizer.json, tokenizer_config.json and special_tokens_map.json from pinned encoder revision ba5fa28ae1361f8115630e252ac49535ffe3bab1 in a named directory below published_baseline_results. Do not copy authentication tokens. A checkpoint filename alone does not verify the model-to-evaluation link. Resolve v2 warnings against completion/evaluation records.

Confirm redistribution terms for original code, models and each source dataset. This support package assigns no new blanket licence. Keep citation and licence notices with upstream components. The revised bundle remains a private release candidate until an actual public deposit/tag/DOI is issued and checked. No manuscript availability claim has been upgraded.
