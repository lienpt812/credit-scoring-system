from __future__ import annotations

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    checking_account: str = Field(alias="Tài khoản vãng lai (USD)")
    duration_months: int = Field(alias="Thời hạn vay (tháng)", ge=1, le=120)
    credit_history: str = Field(alias="Lịch sử tín dụng")
    loan_purpose: str = Field(alias="Mục đích vay")
    savings_account: str = Field(alias="Tài khoản tiết kiệm (USD)")
    experience_years: str = Field(alias="Số năm kinh nghiệm")
    gender: str = Field(alias="Giới tính")
    age_group: str = Field(alias="Tuổi")
    housing: str = Field(alias="Tình trạng nhà ở")
    occupation: str = Field(alias="Nghề nghiệp")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    default_probability: float
    risk_band: str
    recommendation: str
    top_risk_factors: list[str]
    model: str
