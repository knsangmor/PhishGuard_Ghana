# Third-party sources and release status

Review date: 29 October 2026. This notice records source-specific terms and unresolved questions. It does not assign a blanket licence to the project, certify all upstream rights, or establish that a public release has occurred.

## Malicious URLs dataset

Source: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset/data

The source page supplied by the researcher labels the dataset CC0: Public Domain. Evidence: the researcher supplied screenshots of the dataset card and the linked CC0 deed. Licence reference: https://creativecommons.org/publicdomain/zero/1.0/

Source file: malicious_phish.csv. Recorded SHA256: d83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01.

The project samples the source, maps benign to 0 and phishing/malware/defacement to 1, removes conflicts and duplicates, audits encoded-input overlap, and partitions by domain. Generated assumed-benign URLs are a separate augmentation source. These modifications do not constitute endorsement by the dataset compiler. The source aggregates other sources; the posted CC0 label is not independent verification of every upstream right.

## Mendeley holdout

Hannousse, Abdelhakim; Yahiouche, Salima (2021), Web page phishing detection, Mendeley Data, Version 3. https://doi.org/10.17632/c2gw7fy2j4.3

Source: https://data.mendeley.com/datasets/c2gw7fy2j4/3

The official source page lists Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

Recorded source CSV SHA256: 21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c.

The project uses URL strings and status labels, ignores the supplied feature columns, maps legitimate to 0 and phishing to 1, and applies recorded conflict, duplication and overlap exclusions against frozen v3 and PhiUSIIL inputs. The retained holdout comprises 7,659 URLs. Preserve attribution, the source and licence links, and this modification notice when distributing the adapted data. The source authors do not endorse this project.

## PhiUSIIL

Prasad, Arvind; Chandra, Shalini. PhiUSIIL Phishing URL (Website), UCI Machine Learning Repository, dataset 967. The current repository citation dates the dataset 2024; the associated article DOI contains 2023. These dates should not be treated as interchangeable bibliographic evidence.

Source: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Associated article: PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. https://doi.org/10.1016/j.cose.2023.103545

The official UCI page lists CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

Recorded source CSV SHA256: a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6.

The project uses URL strings, reverses the source binary coding to project benign=0/phishing=1, and applies recorded filtering, overlap exclusion and development sampling. Following initial inspection, PhiUSIIL was used as development data; it is not an untouched external confirmation set for revised models. Preserve attribution, licence links and modification notices for redistributed adaptations.

## DomURLs-BERT: redistribution unresolved

Model: amahdaouy/DomURLs_BERT. Pinned revision: ba5fa28ae1361f8115630e252ac49535ffe3bab1.

Model source: https://huggingface.co/amahdaouy/DomURLs_BERT/tree/ba5fa28ae1361f8115630e252ac49535ffe3bab1

Authors' repository: https://github.com/AbdelkaderMH/DomURLs_BERT

Paper: El Mahdaouy et al., DomURLs_BERT: Pre-trained BERT-based Model for Malicious Domains and URLs Detection and Classification. https://doi.org/10.1007/s10922-025-10010-9 ; preprint https://arxiv.org/abs/2409.09143

No explicit redistribution licence was established from the repository and checkpoint pages inspected. This is an unresolved permission question, not a finding that no licence can exist. Download availability and a paper citation do not resolve it.

The project fine-tunes the encoder and adds a classification head. Its full best_model.pt files contain encoder weights. Keep the full historical candidate privately. Until applicable permission is established, the proposed public package should exclude base weights, full fine-tuned DomURLs checkpoints, resume/optimizer checkpoints, and copied upstream tokenizer/config assets. Review adapted preprocessing code for upstream obligations as well. Preserve protocol metadata, hashes, permitted results and an accurate description of any withheld components. The public package must not be described as including all trained checkpoints if these are withheld.

## Other items still requiring decisions

- The exact historical Tranco list used for augmentation and popularity analysis needs its own source/version and terms review. CC0 or CC BY labels above do not apply automatically to it.
- Author-created code and documentation require an explicit licence decision by the relevant rights holders. This notice does not grant that permission or select a licence on their behalf.
- Third-party libraries remain under their own licences; this notice is not a replacement for a dependency licence inventory.
- A separate public-candidate builder must enforce any exclusions. Merely adding this notice does not remove files from an existing ZIP.
- Review the manuscript availability statement against what is actually deposited. A public DOI or release tag remains pending.

No source files, checkpoints or existing candidate archives were modified in preparing this notice.
