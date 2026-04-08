from typing import Tuple, Dict, Any, List
from models import Observation, Action

class Grader:
    def evaluate(self, obs: Observation, action: Action, expected_decisions: Dict[str, str], expected_ranking: List[str]) -> Tuple[float, Dict[str, Any]]:
        total_reward = 0.0
        info = {
            "trial_scores": {},
            "ranking_correct": False,
            "reward_breakdown": {}
        }

        action_evals = {te.trial_id: te for te in action.trial_evaluations}

        # Weighting: Since there are multiple trials, individual rewards are scaled 
        # so total falls nicely in [0,1]. We dedicate 80% to criteria/decision accuracy, 20% to ranking.
        trial_max_base = 0.8 / max(len(obs.trials), 1)

        for trial in obs.trials:
            trial_reward = 0.0
            t_id = trial.id
            exp_dec = expected_decisions.get(t_id, "needs_review")
            act_eval = action_evals.get(t_id)

            if not act_eval:
                info["trial_scores"][t_id] = 0.0
                continue

            # 1. Decision Reward (+0.30 equivalent, scaled)
            if act_eval.decision == exp_dec:
                trial_reward += (trial_max_base * 0.3)

            # 2. Establish Ground Truth for Criteria
            ground_truth_criteria = {}
            ground_truth_criteria["age"] = trial.age_min <= obs.patient.age <= trial.age_max
            for cond in trial.required_conditions:
                ground_truth_criteria[f"requires_{cond}"] = (cond in obs.patient.conditions)
            for cond in trial.excluded_conditions:
                ground_truth_criteria[f"excludes_{cond}"] = (cond not in obs.patient.conditions)
            for lab, criteria_str in trial.lab_criteria.items():
                lab_met = False
                if lab in obs.patient.lab_results:
                    val = obs.patient.lab_results[lab]
                    if ">" in criteria_str:
                        thresh = float(criteria_str.replace(">", "").strip())
                        lab_met = val > thresh
                    elif "<" in criteria_str:
                        thresh = float(criteria_str.replace("<", "").strip())
                        lab_met = val < thresh
                ground_truth_criteria[f"lab_{lab}"] = lab_met

            # 3. Grade the Criteria Evaluations
            action_criteria_dict = {ce.criterion_name: ce.met for ce in act_eval.criterion_evaluations}
            
            crit_weight = (trial_max_base * 0.5) / max(len(ground_truth_criteria), 1) if ground_truth_criteria else 0
            dangerous_penalties = 0.0
            
            for name, gt_met in ground_truth_criteria.items():
                if name in action_criteria_dict:
                    act_met = action_criteria_dict[name]
                    if act_met == gt_met:
                        trial_reward += crit_weight
                    elif name.startswith("excludes_") and gt_met is False and act_met is True:
                        dangerous_penalties -= (trial_max_base * 1.0) # strong penalty portioned to trial max

            trial_reward += dangerous_penalties
            trial_reward = max(0.0, trial_reward) # bind at zero per trial
            
            info["trial_scores"][t_id] = trial_reward
            total_reward += trial_reward

        info["reward_breakdown"]["trials_aggregated"] = total_reward

        # Ranking score: up to +20% overall reward if ranking order matches exactly
        if expected_ranking and action.ranked_trial_ids == expected_ranking:
            total_reward += 0.20
            info["ranking_correct"] = True
            info["reward_breakdown"]["ranking"] = 0.20
        else:
            info["reward_breakdown"]["ranking"] = 0.0

        # Final clip
        total_reward = max(0.0, min(1.0, total_reward))
        info["total_reward"] = total_reward
        
        return total_reward, info
