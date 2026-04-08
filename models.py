from pydantic import BaseModel, Field
from typing import List, Dict, Literal

class Patient(BaseModel):
    age: int
    gender: str
    conditions: List[str]
    lab_results: Dict[str, float]

class TrialCriteria(BaseModel):
    id: str = "T1"
    age_min: int
    age_max: int
    required_conditions: List[str]
    excluded_conditions: List[str]
    lab_criteria: Dict[str, str]

class Observation(BaseModel):
    task_id: str
    patient: Patient
    trials: List[TrialCriteria]

class CriterionEvaluation(BaseModel):
    criterion_name: str
    met: bool
    reason: str

class TrialEvaluation(BaseModel):
    trial_id: str
    decision: Literal["eligible", "ineligible", "needs_review"]
    criterion_evaluations: List[CriterionEvaluation]

class Action(BaseModel):
    reasoning_summary: str = Field(default="No reasoning provided.")
    trial_evaluations: List[TrialEvaluation]
    ranked_trial_ids: List[str] = Field(default_factory=list)
    confidence: float
