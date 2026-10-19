# Member D validation report support

**9,042 validation records** include 1,058 subscribers (11.70%). Threshold-selected F1 values are development results. Final test evaluation awaits explicit approval. CA6001 report support follows.

## Methodology paragraph

We evaluated existing train-fitted Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), MLP and prior-probability Dummy pipelines without retraining. Configurations and the recommended family remained those selected by training cross-validation Average Precision (AP). Validation row identities, labels, alignment, positive-class order (`yes=1`), predictions and pipeline identities were verified. Primary metric AP was complemented by ROC-AUC; AP is recall-increment-weighted precision, not trapezoidal PR area. Precision, recall, F1, accuracy, balanced accuracy, confusion counts, Brier score and log loss were also reported. [AP definition](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html).

For each real model, we maximized unweighted validation F1 over thresholds 0, 1 and the midpoints between distinct scores, preserving tied scores. Exact F1 ties selected the largest candidate. Decisions used `p_yes >= threshold`; Dummy stayed at 0.5. This convention balances precision and recall without assuming unobserved campaign costs, but F1 excludes true negatives and does not optimize profit. Threshold choice changes decisions while leaving score-based metrics unchanged. Full precision is retained in the [approval-pending protocol](../config/final_protocol.proposed.json); displayed thresholds are rounded and must not replace its values. Validation reuse makes selected-threshold performance optimistic for generalization. [Threshold methodology](https://scikit-learn.org/1.8/modules/classification_threshold.html).

## Results paragraph and comparison tables

RF had the highest validation AP (0.3865), ROC-AUC (0.7438) and selected-threshold F1 (0.4209), consistent with its training-CV recommendation. These comparisons establish neither statistical superiority nor final-test performance. MLP had the lowest Brier score and log loss; RF's stronger ranking did not imply better probability estimates. RF used balanced training weights; no scores were assumed calibrated.

| Model | AP | ROC-AUC | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|
| Dummy | 0.1170 | 0.5000 | 0.1033 | 0.3609 |
| LR | 0.3589 | 0.7210 | 0.0894 | 0.3152 |
| DT | 0.3698 | 0.7248 | 0.0884 | 0.3408 |
| RF | 0.3865 | 0.7438 | 0.1765 | 0.5410 |
| MLP | 0.3772 | 0.7376 | 0.0876 | 0.3092 |

“Selected” denotes the F1-optimized rule; predicted-positive counts represent simulated outreach volume at natural validation prevalence.

| Model / rule | Threshold | Precision | Recall | F1 | Accuracy | Balanced accuracy | Predicted positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy / both | 0.500000 | 0.0000* | 0.0000 | 0.0000 | 0.8830 | 0.5000 | 0 |
| LR / 0.5 | 0.500000 | 0.7036 | 0.1682 | 0.2715 | 0.8944 | 0.5794 | 253 |
| LR / selected | 0.167963 | 0.3422 | 0.4140 | 0.3747 | 0.8383 | 0.6543 | 1,280 |
| DT / 0.5 | 0.500000 | 0.7023 | 0.1739 | 0.2788 | 0.8947 | 0.5821 | 262 |
| DT / selected | 0.258660 | 0.4529 | 0.3639 | 0.4036 | 0.8741 | 0.6528 | 850 |
| RF / 0.5 | 0.500000 | 0.2908 | 0.5595 | 0.3827 | 0.7888 | 0.6893 | 2,036 |
| RF / selected | 0.611242 | 0.4270 | 0.4149 | 0.4209 | 0.8664 | 0.6706 | 1,028 |
| MLP / 0.5 | 0.500000 | 0.6891 | 0.1739 | 0.2777 | 0.8942 | 0.5818 | 267 |
| MLP / selected | 0.189610 | 0.3965 | 0.4112 | 0.4037 | 0.8579 | 0.6641 | 1,097 |

*Dummy precision has no positive-prediction denominator; the shared overall metric convention returns zero. Subgroup tables instead leave undefined rates missing.

| Model, selected rule | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Dummy | 7,984 | 0 | 1,058 | 0 |
| LR | 7,142 | 842 | 620 | 438 |
| DT | 7,519 | 465 | 673 | 385 |
| RF | 7,395 | 589 | 619 | 439 |
| MLP | 7,322 | 662 | 623 | 435 |

At 0.5, LR/DT/MLP favored precision but found only approximately 17% of subscribers. Lower thresholds increased recall at the cost of precision and outreach volume. RF moved in the opposite direction: raising its threshold removed 855 false positives while losing 153 true positives. F1 improved, while recall and balanced accuracy fell. DT's selected rule offered greater precision and fewer contacts than RF, but missed more subscribers. Dummy's 88.30% accuracy alongside zero recall shows why accuracy alone is inadequate.

Sources: [metrics](../results/member_d/evaluation/metrics.csv), [complete threshold search](../results/member_d/evaluation/threshold_sweep.csv), [ROC/PR figure](../results/member_d/evaluation/figures/roc_pr.png), [baseline and selected confusion matrices](../results/member_d/evaluation/figures/confusion_matrices.png).

## Error analysis paragraph

Fixed groups revealed heterogeneous RF errors. Counts and within-group rates answer different questions. Groups from different features overlap and must not be summed.

| RF validation group | Records | FN / actual positives | FP / actual negatives |
|---|---:|---:|---:|
| No prior contact | 7,349 | 535 / 669 (79.97%) | 273 / 6,680 (4.09%) |
| Prior contact | 1,693 | 84 / 389 (21.59%) | 316 / 1,304 (24.23%) |
| Previous outcome: success | 312 | 0 / 209 (0.00%) | 103 / 103 (100.00%) |
| Housing loan: yes | 5,032 | 329 / 378 (87.04%) | 37 / 4,654 (0.80%) |
| Housing loan: no | 4,010 | 290 / 680 (42.65%) | 552 / 3,330 (16.58%) |
| Age 30–39 | 3,627 | 273 / 374 (72.99%) | 167 / 3,253 (5.13%) |
| Age ≥60 | 364 | 23 / 128 (17.97%) | 125 / 236 (52.97%) |

The no-prior-contact group contained 535 of RF's 619 false negatives (86.43%). Every record with previous success was selected, capturing all 209 subscribers but also all 103 non-subscribers. Older records had lower FNR but much higher FPR than ages 30–39. These are associations involving different group prevalence and feature combinations; they neither identify causes nor establish fairness. Among negative-balance records, apparent precision of 100% came from just one selected record, while 29 of 30 subscribers were missed—an illustration of why denominators matter.

RF and LR disagreed on 576 records: RF alone was correct on 415, LR alone on 161, and both missed 569 subscribers. RF and DT disagreed on 370: DT alone was correct on 220 and RF alone on 150, despite RF's higher F1. Neither family dominates every decision objective; this does not justify an unplanned ensemble.

Sources: [all 11 feature groups](../results/member_d/errors/error_groups.csv), [pairwise disagreements](../results/member_d/errors/model_disagreements.csv), [row outcomes](../results/member_d/errors/row_errors.csv), [error figure](../results/member_d/errors/figures/error_groups.png). Small denominators below 30 are flagged; group comparisons are exploratory without multiplicity adjustment.

## Explainability paragraph

Permutation importance used complete fitted pipelines, all validation rows, AP scoring, ten repetitions and seed 42. Raw columns were shuffled independently without fitting. The three largest mean AP decreases appear below with shuffle standard deviations (`ddof=0`). [Permutation method](https://scikit-learn.org/1.8/modules/permutation_importance.html).

| Model | Largest | Second | Third |
|---|---|---|---|
| LR | poutcome 0.1092 ± 0.0046 | pdays 0.0849 ± 0.0040 | housing 0.0427 ± 0.0048 |
| DT | poutcome 0.1732 ± 0.0031 | age 0.0516 ± 0.0020 | housing 0.0386 ± 0.0049 |
| RF | poutcome 0.1077 ± 0.0043 | housing 0.0515 ± 0.0056 | age 0.0287 ± 0.0022 |
| MLP | poutcome 0.1038 ± 0.0053 | pdays 0.0992 ± 0.0048 | housing 0.0405 ± 0.0052 |

Previous campaign outcome was the largest measured reliance for all four models. `pdays` represents only the derived prior-contact flag, not exact elapsed days. Dummy's importance was exactly zero for all 11 fields. Small negative importance values were retained, including LR's `previous` mean of −0.0010; they do not establish that removing a feature would improve a retrained model. Correlated history variables can substitute for one another, and independent shuffles may create implausible combinations. Importance measures model dependence, not causal effects. Shuffle SD is not a confidence interval.

Sources: [all feature summaries](../results/member_d/importance/permutation_summary.csv), [individual repetitions](../results/member_d/importance/permutation_repeats.csv), [importance figure](../results/member_d/importance/figures/permutation_importance.png).

## Discussion paragraph

False positives imply outreach to recorded non-subscribers; false negatives imply excluding recorded subscribers. RF's selected rule reduces outreach relative to its default while missing more subscribers, so its business attractiveness depends on costs, capacity and benefits absent from this dataset. Observed subscription is not incremental response caused by contact; monetary gains or losses cannot be inferred. Random record splits do not establish future-period performance or customer isolation. Historical data, uncertain as-of availability, unknown customer identifiers, uncalibrated scores, threshold-selection optimism and exploratory subgroup comparisons constrain generalization. The original training-time source snapshot is unavailable; later verification does not reconstruct it. Prospective validation, identifiable customer grouping, calibration assessment and cost-informed decisions are future studies, not additions to this frozen experiment.

## Conclusion paragraph

The validation evidence supports retaining RF as the training-CV-recommended family while showing that operating thresholds materially alter precision, recall and outreach volume. All real models ranked above Dummy, but their probability-quality and decision trade-offs differed. The proposed thresholds are ready for approval; final generalization conclusions require the authorized one-shot test with unchanged pipelines and thresholds.

## Course integration

Preserve ten template sections and the ten-page body limit, including references, excluding cover. Integrate methodology, selected comparison tables/ROC/PR, and error/importance findings into the corresponding sections. Detailed CSVs support reproducibility, without exempting report content from the page limit. APA-style references and procedures are in [Member D methods](MEMBER_D_METHODS.md); retain the repository's UCI citation and notices.
