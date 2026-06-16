from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, FIGURE_DIR, METRICS_PATH, MODEL_PATH, MODEL_REPORT_PATH
from src.data import load_train_test, split_features_target
from src.features import build_preprocessor


def train_and_evaluate(
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
) -> dict[str, Any]:
    train_df, test_df = load_train_test()
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=4,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    results: dict[str, Any] = {}
    best_name = ""
    best_score = -1.0
    best_pipeline: Pipeline | None = None

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        threshold_table = evaluate_thresholds(y_test, probabilities)
        best_threshold = max(threshold_table, key=lambda row: row["business_score"])["threshold"]
        predictions = (probabilities >= best_threshold).astype(int)
        metrics = classification_metrics(y_test, predictions, probabilities)
        metrics["selected_threshold"] = best_threshold
        metrics["threshold_table"] = threshold_table
        results[name] = metrics

        ranking_score = metrics["roc_auc"] + metrics["average_precision"]
        if ranking_score > best_score:
            best_score = ranking_score
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was trained.")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipeline, "features": FEATURE_COLUMNS, "best_model": best_name}, model_path)

    report = {
        "target": "1 = bad credit, 0 = good credit",
        "best_model": best_name,
        "models": results,
        "test_rows": len(test_df),
        "train_rows": len(train_df),
    }
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_report_figures(report, FIGURE_DIR)
    save_model_report(report, MODEL_REPORT_PATH)
    return report


def classification_metrics(y_true, y_pred, probabilities) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "precision_bad": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_bad": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_bad": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": matrix,
    }


def evaluate_thresholds(y_true, probabilities) -> list[dict[str, Any]]:
    rows = []
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        approval_rate = round(float((tn + fn) / len(y_true)), 4)
        bad_recall = round(float(tp / max(1, tp + fn)), 4)
        business_score = round(float(tp * 5 - fp * 1 - fn * 8), 4)
        rows.append(
            {
                "threshold": threshold,
                "bad_recall": bad_recall,
                "approval_rate": approval_rate,
                "false_approve_bad": int(fn),
                "false_reject_good": int(fp),
                "business_score": business_score,
            }
        )
    return rows


def save_report_figures(report: dict[str, Any], figure_dir: str | Path = FIGURE_DIR) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_model_comparison_figure(report, figure_dir / "model_comparison.png")
    save_threshold_tuning_figure(report, figure_dir / "threshold_tuning.png")
    save_confusion_matrix_figure(report, figure_dir / "confusion_matrix.png")


def save_model_comparison_figure(report: dict[str, Any], output_path: Path) -> None:
    model_names = list(report["models"].keys())
    roc_auc = [report["models"][name]["roc_auc"] for name in model_names]
    average_precision = [report["models"][name]["average_precision"] for name in model_names]
    x_positions = range(len(model_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x - width / 2 for x in x_positions], roc_auc, width, label="ROC-AUC", color="#2563eb")
    ax.bar(
        [x + width / 2 for x in x_positions],
        average_precision,
        width,
        label="Average precision",
        color="#16a34a",
    )
    ax.set_title("Model comparison on holdout test set")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x_positions), [name.replace("_", "\n") for name in model_names])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_threshold_tuning_figure(report: dict[str, Any], output_path: Path) -> None:
    best_model = report["best_model"]
    rows = report["models"][best_model]["threshold_table"]
    thresholds = [row["threshold"] for row in rows]
    business_scores = [row["business_score"] for row in rows]
    bad_recall = [row["bad_recall"] for row in rows]
    approval_rate = [row["approval_rate"] for row in rows]

    fig, ax_score = plt.subplots(figsize=(9, 5))
    ax_score.plot(thresholds, business_scores, marker="o", color="#dc2626", label="Business score")
    ax_score.set_title(f"Threshold tuning for {best_model.replace('_', ' ')}")
    ax_score.set_xlabel("Decision threshold")
    ax_score.set_ylabel("Business score")
    ax_score.grid(alpha=0.25)

    ax_rates = ax_score.twinx()
    ax_rates.plot(thresholds, bad_recall, marker="s", color="#2563eb", label="Bad recall")
    ax_rates.plot(thresholds, approval_rate, marker="^", color="#16a34a", label="Approval rate")
    ax_rates.set_ylabel("Rate")
    ax_rates.set_ylim(0, 1)

    lines, labels = ax_score.get_legend_handles_labels()
    rate_lines, rate_labels = ax_rates.get_legend_handles_labels()
    ax_score.legend(lines + rate_lines, labels + rate_labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_figure(report: dict[str, Any], output_path: Path) -> None:
    best_model = report["best_model"]
    matrix = report["models"][best_model]["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion matrix: {best_model.replace('_', ' ')}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], ["Good", "Bad"])
    ax.set_yticks([0, 1], ["Good", "Bad"])

    max_value = max(max(row) for row in matrix)
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            color = "white" if value > max_value / 2 else "#111827"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color, fontsize=13)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_model_report(report: dict[str, Any], output_path: str | Path = MODEL_REPORT_PATH) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_model_report_markdown(report), encoding="utf-8")


def build_model_report_markdown(report: dict[str, Any]) -> str:
    best_model = report["best_model"]
    best_metrics = report["models"][best_model]
    selected_threshold = best_metrics["selected_threshold"]
    tn, fp = best_metrics["confusion_matrix"][0]
    fn, tp = best_metrics["confusion_matrix"][1]
    best_threshold_row = next(
        row for row in best_metrics["threshold_table"] if row["threshold"] == selected_threshold
    )

    model_rows = [
        "| Model | ROC-AUC | Average Precision | Precision Bad | Recall Bad | F1 Bad | Selected Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, metrics in report["models"].items():
        marker = " (selected)" if model_name == best_model else ""
        model_rows.append(
            "| "
            f"{model_name}{marker} | "
            f"{metrics['roc_auc']:.4f} | "
            f"{metrics['average_precision']:.4f} | "
            f"{metrics['precision_bad']:.4f} | "
            f"{metrics['recall_bad']:.4f} | "
            f"{metrics['f1_bad']:.4f} | "
            f"{metrics['selected_threshold']:.1f} |"
        )

    threshold_rows = [
        "| Threshold | Bad Recall | Approval Rate | False Approve Bad | False Reject Good | Business Score |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_metrics["threshold_table"]:
        threshold_rows.append(
            "| "
            f"{row['threshold']:.1f} | "
            f"{row['bad_recall']:.4f} | "
            f"{row['approval_rate']:.4f} | "
            f"{row['false_approve_bad']} | "
            f"{row['false_reject_good']} | "
            f"{row['business_score']:.1f} |"
        )

    return "\n".join(
        [
            "# Credit Scoring Model Report",
            "",
            "## Executive Summary",
            "",
            f"- Training rows: {report['train_rows']}",
            f"- Test rows: {report['test_rows']}",
            f"- Selected model: `{best_model}`",
            f"- Selected decision threshold: `{selected_threshold:.1f}`",
            f"- ROC-AUC: `{best_metrics['roc_auc']:.4f}`",
            f"- Average Precision: `{best_metrics['average_precision']:.4f}`",
            f"- Recall for bad credit customers: `{best_metrics['recall_bad']:.4f}`",
            "",
            "The selected model is chosen by ranking ROC-AUC plus Average Precision. This balances broad ranking quality with performance on the bad-credit class, which is the minority and higher-risk class.",
            "",
            "## Model Comparison",
            "",
            *model_rows,
            "",
            "Interpretation:",
            "",
            "- ROC-AUC measures how well the model ranks bad-credit customers above good-credit customers across thresholds. Higher is better.",
            "- Average Precision summarizes precision-recall performance and is useful when the bad-credit class is relatively rare. Higher is better.",
            "- Recall Bad answers: among truly bad-credit customers, how many did the model catch?",
            "- Precision Bad answers: among customers flagged as bad credit, how many were actually bad?",
            "- F1 Bad balances precision and recall for the bad-credit class.",
            "",
            "## Threshold Tuning",
            "",
            *threshold_rows,
            "",
            "The final threshold is selected by the highest business score in the threshold table. In this project, missing a bad customer is treated as more costly than rejecting a good customer, so the selected threshold favors catching bad-credit customers even if some good customers are sent to review or rejected.",
            "",
            f"At threshold `{selected_threshold:.1f}`, the selected model reaches bad-customer recall of `{best_threshold_row['bad_recall']:.4f}` and an approval rate of `{best_threshold_row['approval_rate']:.4f}`.",
            "",
            "## Confusion Matrix Explanation",
            "",
            "| Actual / Predicted | Predicted Good | Predicted Bad |",
            "|---|---:|---:|",
            f"| Actual Good | {tn} | {fp} |",
            f"| Actual Bad | {fn} | {tp} |",
            "",
            "- Actual Good / Predicted Good: good customers correctly approved.",
            "- Actual Good / Predicted Bad: good customers incorrectly rejected or sent to manual review.",
            "- Actual Bad / Predicted Good: bad customers incorrectly approved. This is the riskiest error.",
            "- Actual Bad / Predicted Bad: bad customers correctly caught.",
            "",
            "## Generated Artifacts",
            "",
            "- `models/credit_scoring_model.joblib`: trained sklearn pipeline and metadata.",
            "- `reports/metrics.json`: machine-readable metrics for all candidate models.",
            "- `reports/model_report.md`: human-readable interpretation of the training result.",
            "- `reports/figures/model_comparison.png`: visual comparison of model ranking metrics.",
            "- `reports/figures/threshold_tuning.png`: threshold trade-off chart for the selected model.",
            "- `reports/figures/confusion_matrix.png`: confusion matrix for the selected model.",
            "",
            "## Notes",
            "",
            "This report is generated automatically by `python -m src.train`, so it should be regenerated whenever the data, features, model candidates, or threshold policy changes.",
            "",
        ]
    )


def load_model(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(model_path)


def predict_risk(input_df: pd.DataFrame, model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    bundle = load_model(model_path)
    pipeline: Pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(input_df[FEATURE_COLUMNS])[:, 1][0])
    band = risk_band(probability)
    return {
        "default_probability": round(probability, 4),
        "risk_band": band,
        "recommendation": recommendation_for_band(band),
        "top_risk_factors": top_risk_factors(input_df.iloc[0].to_dict()),
        "model": bundle.get("best_model", "unknown"),
    }


def risk_band(probability: float) -> str:
    if probability >= 0.6:
        return "High Risk"
    if probability >= 0.3:
        return "Medium Risk"
    return "Low Risk"


def recommendation_for_band(band: str) -> str:
    if band == "High Risk":
        return "Reject or manual review"
    if band == "Medium Risk":
        return "Manual review"
    return "Approve"


def top_risk_factors(row: dict[str, Any]) -> list[str]:
    factors: list[str] = []
    if row.get("Tài khoản vãng lai (USD)") in {"<0", "Không có"}:
        factors.append("Low or missing checking account balance")
    if row.get("Tài khoản tiết kiệm (USD)") in {"Ít", "Không có"}:
        factors.append("Low savings account balance")
    if int(row.get("Thời hạn vay (tháng)", 0)) >= 36:
        factors.append("Long loan duration")
    if row.get("Lịch sử tín dụng") in {"Không có", "Chậm trả trước đây"}:
        factors.append("Weak or missing credit history")
    if not factors:
        factors.append("No major rule-based risk factor detected")
    return factors[:3]
