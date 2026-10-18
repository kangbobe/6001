# CA6001 Technical Report — Working Outline

> Draft scaffold only. Replace bracketed prompts with verified project details and results. No model result is asserted here.

## Cover Page (excluded from the 10-page body limit)

- Project title: [Insert final title]
- Student names: [Insert names]

## 1. Introduction

- Problem domain and motivation: [Describe the historical bank-marketing prediction context.]
- Objective: [State the record-level subscription prediction task.]
- Scope and interpretation: [Clarify the offline retrospective screening simulation and its limits.]

## 2. Problem Definition

- Formal task: [Binary classification; define `yes=1` and the prediction unit.]
- Constraints and assumptions: [State input availability assumptions and the limits of the available time/customer identifiers.]
- Evaluation question: [Insert the prespecified comparison question.]

## 3. Dataset Description

- Source, version, license, and format: [Cite the verified UCI legacy Bank Marketing source and local integrity record.]
- Dataset structure and split: [Insert verified counts from the saved data audit and split manifest.]
- Input policy: [List the 11 source fields; explain removal of duration, campaign, contact, day, and month.]
- Preprocessing: [Explain `pdays` to `has_previous_contact=(pdays>=0)`, removal of exact elapsed days, retained `unknown`, and retained extreme/negative values.]
- Exploratory analysis: [Select train-only target, age/balance, unknown, job/poutcome, and contact-history diagnostics.]

## 4. Algorithm Selection and Rationale

- Candidate models: [Logistic Regression, Decision Tree, Random Forest, MLP, and Dummy baseline.]
- Theory and suitability: [Briefly compare linear, single-tree, ensemble, and neural approaches for this feature/task setup.]
- Strengths, limitations, and alternatives: [Use sourced, balanced descriptions; do not claim an empirically best model here.]

## 5. Methodology

- Data and feature processing: [Describe the shared preprocessing pipeline and fold-local fitting.]
- Training and model selection: [Insert the fixed train-only stratified CV protocol, primary Average Precision metric, search spaces, tie rules, and class-weight handling.]
- Validation and threshold boundary: [Explain that validation supports development diagnostics and D's threshold choice.]
- Final evaluation protocol: [D to document the frozen model hashes, thresholds, decision rule, and one-shot test procedure.]
- Tools and environment: [Insert exact versions from the run record.]

## 6. Implementation

- Core functions and design decisions: [Summarize data loading, preprocessing, fold evaluation, and export interfaces.]
- Main algorithm excerpt: [Insert a short, representative code fragment.]
- Reproducibility artifacts: [List manifests, model pipelines, validation predictions, and logs.]

## 7. Results and Evaluation

- Cross-validation development results: [Insert verified per-model CV results and variability; distinguish fold standard deviation from confidence intervals.]
- Frozen test comparison: [D to insert metrics and threshold-specific confusion counts after the authorized one-shot evaluation.]
- Figures and tables: [Insert selected model comparisons, PR/ROC curves, and confusion summaries.]
- Error analysis: [D to report validation/test error groups under the frozen protocol and state the analysis population.]
- Do not write numerical claims until the corresponding output files exist and have been read back.

## 8. Discussion & Insights

- Findings: [D to interpret verified results without turning association into causation.]
- Limitations: [Discuss retrospective record-level evaluation, unknown customer grouping, time/as-of uncertainty, data vintage, and model assumptions as supported by evidence.]
- Improvements: [Propose bounded next steps tied to observed evidence.]

## 9. Conclusion

- Main outcome: [D to summarize the verified comparison.]
- Key insight and reflection: [D to connect the task definition, algorithm choices, implementation, and limitations.]

## 10. References

- [Add the dataset, software, methods, and other sources actually used; format in APA style.]
