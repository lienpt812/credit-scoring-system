from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "train.xlsx"
TEST_PATH = ROOT_DIR / "test.xlsx"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_PATH = MODEL_DIR / "credit_scoring_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
MODEL_REPORT_PATH = REPORT_DIR / "model_report.md"

TARGET_COLUMN = "Hạng tín dụng của KH"
BAD_LABEL = "Xấu"
GOOD_LABEL = "Tốt"

NUMERIC_FEATURES = ["Thời hạn vay (tháng)"]
CATEGORICAL_FEATURES = [
    "Tài khoản vãng lai (USD)",
    "Lịch sử tín dụng",
    "Mục đích vay",
    "Tài khoản tiết kiệm (USD)",
    "Số năm kinh nghiệm",
    "Giới tính",
    "Tuổi",
    "Tình trạng nhà ở",
    "Nghề nghiệp",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
