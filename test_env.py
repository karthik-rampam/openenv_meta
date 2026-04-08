from environment import ClinicalTrialEnv
from models import Action

def test_environment():
    try:
        env = ClinicalTrialEnv()
        print(f"Success: Loaded {len(env.cases)} cases.")
        
        # Test 1: Reset
        print("\n--- Testing reset() ---")
        obs = env.reset()
        print(f"Observation task_id: {obs.task_id}")
        expected = env.expected_decision
        print(f"Environment expected decision: {expected}")
        print(f"Patient age: {obs.patient.age}, T2D required: {'Type 2 Diabetes' in obs.trial.required_conditions}")

        # Test 2: Step with correct decision
        print("\n--- Testing step() with correct action ---")
        action_correct = Action(
            decision=expected,
            criterion_evaluations=[],
            confidence=1.0
        )
        _, reward, done, info = env.step(action_correct)
        print(f"Reward for correct action: {reward}")
        print(f"Done: {done}, Info: {info}")

        # Test 3: Step with incorrect decision
        print("\n--- Testing step() with incorrect action ---")
        wrong_decision = "eligible" if expected == "ineligible" else "ineligible"
        action_wrong = Action(
            decision=wrong_decision,
            criterion_evaluations=[],
            confidence=0.5
        )
        _, reward2, done2, info2 = env.step(action_wrong)
        print(f"Reward for wrong action: {reward2}")
        print(f"Done: {done2}, Info: {info2}")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    test_environment()
