# Member D evaluation methodology (validation only)

## Data and model boundary

Use the 9,042 frozen validation records, with `yes=1`, in the exact saved row order. Retain all four train-fitted model families and the prior-probability Dummy baseline. Configuration selection and the family recommendation remain based on the existing train-only CV AP. No hyperparameters, features, splits, weights, fitted transformations or model artifacts are changed. Models are not retrained on train+validation.

Before analysis, validate original artifact identities; each prediction file's row IDs, order, uniqueness, split, family, binary labels, finite [0,1] scores and original 0.5 decisions; agreement across families and with validation labels; fitted classes `[0,1]`; score reproduction within 1e-12; and reproduction of original metrics. The D reader materializes only validation lines. The shared CSV is scanned as opaque bytes for identity/line selection, while split files are read as membership metadata. Non-validation CSV records are never decoded or parsed by this reader.

## Threshold procedure, specified before computing D results

For each real model, let the sorted distinct saved validation scores be s1,...,sk. The finite candidate set is `{0, 1}` plus `(si + si+1)/2` for adjacent distinct scores, computed as `si + (si+1-si)/2` for numerical stability. If a midpoint rounds to the lower score, use the upper score. Candidates preserve tied scores and enumerate every decision partition realizable by thresholds within [0,1]. If a score equals 1, predicting everyone negative is not realizable in that threshold range; this does not affect the maximum F1 with positive labels present.

Classify yes when `p_yes >= t`. Compute unweighted `F1 = 2TP/(2TP+FP+FN)` for every candidate. Maximize F1; compare integer numerator/denominator cross-products to detect exact ties, then choose the largest candidate threshold. For distinct decision sets with equal F1 this weakly favors fewer contacts. Dummy remains fixed at 0.5. Retain full threshold precision in JSON; rounded numbers in report tables are display only. Verify the resulting classifications against freshly reloaded fitted pipelines.

This is the handoff's proposed academic F1 convention, chosen without known campaign costs, capacity constraints or causal response values. It is not profit optimization and F1 does not use true negatives. Selecting thresholds on validation makes the reported selected-threshold F1 an in-sample development criterion. The untouched final test is needed for final performance claims.

## Metrics and figures

Report AP (`sklearn.metrics.average_precision_score`) as primary ranking quality and ROC-AUC as complementary discrimination. AP weights precision by recall increments; it is not trapezoidal PR area. Plot full ROC and precision-recall curves, marking selected operating points and the validation prevalence reference. Include Dummy.

At both 0.5 and the selected threshold, report accuracy, balanced accuracy, precision, recall, F1 and TN/FP/FN/TP. Also report Brier score and log loss as score-quality diagnostics; lower is better, but neither is proof of calibration. AP, ROC-AUC, Brier and log loss do not change with the decision threshold. Class imbalance makes accuracy alone misleading. Compare realized positive predictions as an outreach-volume proxy and explain associated FP/FN trade-offs.

## Error analysis

Use observed validation labels and selected predictions. Export each row's TP/TN/FP/FN status and pairwise disagreements, including errors shared and corrected by each family. Group by the seven original categorical fields, keeping literal unknown; fixed age groups <30, 30-39, 40-49, 50-59, >=60; balance <0, =0, 1-999, 1000-4999, >=5000; previous contacts 0, 1, 2-4, >=5; and no prior contact (`pdays=-1`) versus prior contact (`pdays>=0`). These bins are documented before inspecting validation error-group outcomes and are not learned from labels.

Report group n, positive/negative counts, selected count, confusion counts, prevalence, precision, recall/FNR and FPR with the appropriate denominators. Undefined rates are missing. Flag denominators below 30; that is an explicit descriptive caution, not a significance cutoff. Groups overlap across feature families and must not be added together. These exploratory comparisons do not prove fairness, discrimination or causes of errors; small groups and multiple comparisons limit interpretation.

FP means contacting a recorded non-subscriber under the retrospective simulation; FN means excluding a recorded subscriber. Outreach cost and missed opportunities are plausible implications, not observed costs or causal lost revenue. The data do not identify who would subscribe because of contacting them.

## Explainability

Run sklearn permutation importance on each complete fitted pipeline and all validation rows, using AP, 10 repeats, random_state=42, n_jobs=1 and max_samples=1.0, with numerical thread pools limited to one. Permute one raw input column at a time; the pipeline then applies its existing transformations. Save every AP drop and the mean and population SD (`ddof=0`) over repeats, including negative values. Dummy supplies a zero-reliance control. Runtime guards block fitting and verify unchanged fitted objects/artifacts.

Importance is baseline AP minus AP after a shuffled feature. It measures this fitted model's reliance on the feature for this validation distribution, not intrinsic usefulness or causality. `pdays` is interpreted only through the pipeline's derived prior-contact flag; no exact elapsed-day effect is estimated. Correlated history variables can substitute for one another, and independent shuffles can create implausible combinations. Repeat SD measures shuffle variability, not a population confidence interval. No feature is removed on the basis of this diagnostic.

## Reporting limits and course integration

Preserve the ten CA6001 template sections and the 10-page body limit including references, excluding cover. D supplies compact material for Methodology, Implementation, Results and Evaluation, Discussion & Insights, and Conclusion. Detailed CSVs and code are supporting reproducibility artifacts, not assumed exemptions from the report page limit. Final test results remain pending; validation findings cannot be relabeled as test findings. No significance or confidence-interval claim is made from descriptive differences.

Random record splits do not establish future-period performance or customer separation; customer IDs, as-of field availability, historical-source shifts, score calibration, selection optimism, subgroup uncertainty, and the missing training-time source snapshot remain limitations.

## Primary method references

Scikit-learn developers. (n.d.). *Average precision score* (version 1.8 documentation). https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html

Scikit-learn developers. (n.d.). *Tuning the decision threshold for class prediction* (version 1.8 documentation). https://scikit-learn.org/1.8/modules/classification_threshold.html

Scikit-learn developers. (n.d.). *Permutation feature importance* (version 1.8 documentation). https://scikit-learn.org/1.8/modules/permutation_importance.html

The repository's original UCI data citation and notices remain applicable. Documentation checked 19 October 2026; the computation uses the locked package versions recorded in each receipt.
