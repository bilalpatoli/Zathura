from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    detrend_window: int = Field(default=301, ge=101, le=2001, strict=True)
    outlier_sigma: float = Field(default=5.0, ge=2, le=10)
    period_grid_size: int = Field(default=2000, ge=1000, le=50000, strict=True)
    detection_threshold: float = Field(default=6.0, ge=0, le=100, allow_inf_nan=False)
    duration_min_days: float = Field(default=0.04, ge=0.02, le=0.5)
    duration_max_days: float = Field(default=0.3, ge=0.02, le=0.5)

    @model_validator(mode="after")
    def validate_windows(self):
        if self.detrend_window % 2 != 1:
            raise ValueError("detrend_window must be odd")
        if self.duration_min_days >= self.duration_max_days:
            raise ValueError("duration_min_days must be less than duration_max_days")
        return self


BASELINE_CONFIG = DetectionConfig()


class Source(BaseModel):
    title: str
    url: str
    snippet: str
    mock: bool = False


class ReviewerDecision(BaseModel):
    decision: Literal["approve", "reject", "revise"]
    rationale: str


class EvaluationResult(BaseModel):
    improved_vs_baseline: bool
    improved_vs_best: bool
    delta: float
    delta_vs_best: float | None = None
    delta_vs_baseline_ci_low: float | None = None
    delta_vs_baseline_ci_high: float | None = None
    significant_vs_baseline: bool | None = None
    delta_vs_best_ci_low: float | None = None
    delta_vs_best_ci_high: float | None = None
    significant_vs_best: bool | None = None


class ExperimentRecord(BaseModel):
    experiment_id: str
    hypothesis: str
    cited_sources: list[Source]
    proposed_config: DetectionConfig
    reviewer_decision: ReviewerDecision
    measurements: dict[str, float]
    runtime_s: float
    evaluation_result: EvaluationResult
    memory_refs: list[str] = Field(default_factory=list)
    accepted: bool
    data_source: Literal["real", "placeholder"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
