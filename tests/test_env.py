import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from environment import ClinicalTrialEnv
from models import Action, TrialEvaluation

@pytest.fixture
def env():
    return ClinicalTrialEnv()

def test_environment_loads_all_cases(env):
    assert len(env.cases) > 10, "Should load Easy, Medium, and Hard cases."

def test_reset_yields_observation(env):
    obs = env.reset()
    assert obs.task_id is not None
    assert len(obs.trials) >= 1

def test_step_function_signature_and_bounds(env):
    obs = env.reset()
    
    dummy_evals = [
      TrialEvaluation(trial_id=t.id, decision="needs_review", criterion_evaluations=[]) 
      for t in obs.trials
    ]
    
    action = Action(
        trial_evaluations=dummy_evals,
        ranked_trial_ids=[],
        confidence=0.5
    )
    
    next_obs, reward, done, info = env.step(action)
    
    assert 0.0 <= reward <= 1.0
    assert done is True
    assert "reward_breakdown" in info
