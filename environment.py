import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from models import Observation, Action, Patient, TrialCriteria
from grader import Grader

class ClinicalTrialEnv:
    def __init__(self):
        self.current_state: Optional[Observation] = None
        self.expected_decisions = {}
        self.expected_ranking = []
        self.cases = self._load_cases()
        self.grader = Grader()

    def _load_cases(self) -> list:
        easy_path = Path(__file__).parent / "app" / "data" / "easy_cases.json"
        medium_path = Path(__file__).parent / "app" / "data" / "medium_cases.json"
        hard_path = Path(__file__).parent / "app" / "data" / "hard_cases.json"
        
        cases = []
        for p in [easy_path, medium_path, hard_path]:
            if p.exists():
                with open(p, "r") as f:
                    loaded = json.load(f)
                    for c in loaded:
                        # Backward compatibility normalization for Phase 1-4 JSONs
                        if "trial" in c:
                            c["trials"] = [c["trial"]]
                            c["trials"][0]["id"] = "T1"
                            c["expected_decisions"] = {"T1": c["expected_decision"]}
                            c["expected_ranking"] = ["T1"]
                        # Pre-fill lists even if missing backwards
                        if "expected_decisions" not in c:
                            c["expected_decisions"] = {}
                        if "expected_ranking" not in c:
                            c["expected_ranking"] = []
                    cases.extend(loaded)
        return cases

    def reset(self) -> Observation:
        case = random.choice(self.cases)
        
        patient = Patient(**case["patient"])
        trials = [TrialCriteria(**t) for t in case["trials"]]
        
        self.expected_decisions = case["expected_decisions"]
        self.expected_ranking = case["expected_ranking"]
        
        self.current_state = Observation(
            task_id=case["task_id"],
            patient=patient,
            trials=trials
        )
        return self.current_state

    def state(self) -> Observation:
        if self.current_state is None:
            return self.reset()
        return self.current_state

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        reward, info = self.grader.evaluate(
            obs=self.current_state,
            action=action, 
            expected_decisions=self.expected_decisions,
            expected_ranking=self.expected_ranking
        )
        
        info["expected_decisions"] = self.expected_decisions
        info["expected_ranking"] = self.expected_ranking
        info["actual_ranking"] = action.ranked_trial_ids
        
        done = True
        return self.current_state, reward, done, info
