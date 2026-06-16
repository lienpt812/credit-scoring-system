# Credit Scoring Model Report

## Executive Summary

- Training rows: 1000
- Test rows: 200
- Selected model: `random_forest`
- Selected decision threshold: `0.4`
- ROC-AUC: `0.9461`
- Average Precision: `0.8121`
- Recall for bad credit customers: `1.0000`

The selected model is chosen by ranking ROC-AUC plus Average Precision. This balances broad ranking quality with performance on the bad-credit class, which is the minority and higher-risk class.

## Model Comparison

| Model | ROC-AUC | Average Precision | Precision Bad | Recall Bad | F1 Bad | Selected Threshold |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8587 | 0.6385 | 0.2917 | 0.9459 | 0.4459 | 0.3 |
| random_forest (selected) | 0.9461 | 0.8121 | 0.4253 | 1.0000 | 0.5968 | 0.4 |
| gradient_boosting | 0.9252 | 0.7656 | 0.5397 | 0.9189 | 0.6800 | 0.3 |

Interpretation:

- ROC-AUC measures how well the model ranks bad-credit customers above good-credit customers across thresholds. Higher is better.
- Average Precision summarizes precision-recall performance and is useful when the bad-credit class is relatively rare. Higher is better.
- Recall Bad answers: among truly bad-credit customers, how many did the model catch?
- Precision Bad answers: among customers flagged as bad credit, how many were actually bad?
- F1 Bad balances precision and recall for the bad-credit class.

## Threshold Tuning

| Threshold | Bad Recall | Approval Rate | False Approve Bad | False Reject Good | Business Score |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 1.0000 | 0.1800 | 0 | 127 | 58.0 |
| 0.3 | 1.0000 | 0.3550 | 0 | 92 | 93.0 |
| 0.4 | 1.0000 | 0.5650 | 0 | 50 | 135.0 |
| 0.5 | 0.8919 | 0.7050 | 4 | 26 | 107.0 |
| 0.6 | 0.7568 | 0.8000 | 9 | 12 | 56.0 |
| 0.7 | 0.4595 | 0.8900 | 20 | 5 | -80.0 |

The final threshold is selected by the highest business score in the threshold table. In this project, missing a bad customer is treated as more costly than rejecting a good customer, so the selected threshold favors catching bad-credit customers even if some good customers are sent to review or rejected.

At threshold `0.4`, the selected model reaches bad-customer recall of `1.0000` and an approval rate of `0.5650`.

## Confusion Matrix Explanation

| Actual / Predicted | Predicted Good | Predicted Bad |
|---|---:|---:|
| Actual Good | 113 | 50 |
| Actual Bad | 0 | 37 |

- Actual Good / Predicted Good: good customers correctly approved.
- Actual Good / Predicted Bad: good customers incorrectly rejected or sent to manual review.
- Actual Bad / Predicted Good: bad customers incorrectly approved. This is the riskiest error.
- Actual Bad / Predicted Bad: bad customers correctly caught.

## Generated Artifacts

- `models/credit_scoring_model.joblib`: trained sklearn pipeline and metadata.
- `reports/metrics.json`: machine-readable metrics for all candidate models.
- `reports/model_report.md`: human-readable interpretation of the training result.
- `reports/figures/model_comparison.png`: visual comparison of model ranking metrics.
- `reports/figures/threshold_tuning.png`: threshold trade-off chart for the selected model.
- `reports/figures/confusion_matrix.png`: confusion matrix for the selected model.

## Notes

This report is generated automatically by `python -m src.train`, so it should be regenerated whenever the data, features, model candidates, or threshold policy changes.
