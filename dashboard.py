import os
import json
import random
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from environment import ClinicalTrialEnv
from models import Action, Patient

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("OPENAI_API_KEY", "")

env = ClinicalTrialEnv()

def get_shared_schema():
    return """
You MUST output valid JSON ONLY matching exactly this schema:
{
  "reasoning_summary": "Summarize total batch findings.",
  "trial_evaluations": [
    {
      "trial_id": "string (the ID of the trial OR the ID of the patient)",
      "decision": "eligible" | "ineligible" | "needs_review",
      "reason": "One specific sentence explaining this exact match decision.",
      "criterion_evaluations": [
        {
          "criterion_name": "string",
          "met": true/false,
          "reason": "string"
        }
      ]
    }
  ],
  "ranked_trial_ids": ["id_1", "id_2"],
  "confidence": 0.0 to 1.0
}
"""

def build_diagnostic_prompt(obs):
    patient = obs.patient.model_dump()
    trials = [t.model_dump() for t in obs.trials]
    return f"""
You are an expert Clinical Trial Coordinator AI. 
Evaluate ONE patient against ALL available trials.

PATIENT RECORD:
{json.dumps(patient, indent=2)}

AVAILABLE TRIALS:
{json.dumps(trials, indent=2)}

{get_shared_schema()}
"""

def build_recruitment_prompt(trial, patients):
    # Map for recruitment
    return f"""
You are an expert Recruitment Specialist. 
Evaluate ONE trial against a batch of patients.

TRIAL CRITERIA:
{json.dumps(trial.model_dump(), indent=2)}

PATIENT DATABASE:
{json.dumps([{"id": f"P{i+1}", "data": p.model_dump()} for i, p in enumerate(patients)], indent=2)}

Rule: In "trial_evaluations", use the Patient ID (P1, P2...) as the "trial_id" field.

{get_shared_schema()}
"""

def run_diagnostic():
    obs = env.reset()
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    prompt = build_diagnostic_prompt(obs)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        action = Action(**result)
        _, reward, _, _ = env.step(action)

        p_info = f"**Patient:** {obs.task_id} | **Age:** {obs.patient.age} | **Conditions:** {', '.join(obs.patient.conditions)}"
        eval_md = f"### 🧠 AI Overview\n> {action.reasoning_summary}\n\n**⚖️ Accuracy Score:** {reward*100:.1f}%"
        
        table_data = []
        for t in obs.trials:
            eval_item = next((te for te in result.get("trial_evaluations", []) if te["trial_id"] == t.id), None)
            decision = eval_item["decision"].upper() if eval_item else "PENDING"
            reason = eval_item["reason"] if eval_item else "..."
            icon = "✅" if "ELIGIBLE" in decision else ("❌" if "INELIGIBLE" in decision else "⚠️")
            
            reqs = ", ".join(t.required_conditions) if t.required_conditions else "None"
            table_data.append([t.id, reqs, f"{decision} {icon}", reason])
            
        return p_info, eval_md, table_data
    except Exception as e:
        return f"Error", f"Error: {e}", []

def run_recruitment():
    obs = env.reset()
    trial = obs.trials[0]
    
    # Sample 4 random patients
    raw_cases = env.cases
    random_cases = random.sample(raw_cases, min(4, len(raw_cases)))
    mock_patients = [Patient(**c["patient"]) for c in random_cases]
    patient_ids = [c["task_id"] for c in random_cases]
    
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    prompt = build_recruitment_prompt(trial, mock_patients)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        evals = result.get("trial_evaluations", [])
        
        t_info = f"**Target Trial:** {trial.id} | **Requires:** {', '.join(trial.required_conditions)}"
        eval_md = f"### 📊 Recruitment Summary\n> {result.get('reasoning_summary', 'Batch Complete.')}"
        
        table_data = []
        for i, p_id in enumerate(patient_ids):
            eval_item = next((te for te in evals if te["trial_id"] in [f"P{i+1}", p_id]), None)
            decision = eval_item["decision"].upper() if eval_item else "REVIEW"
            reason = eval_item["reason"] if eval_item else "..."
            icon = "✅" if "ELIGIBLE" in decision else ("❌" if "INELIGIBLE" in decision else "⚠️")
            
            conds = ", ".join(mock_patients[i].conditions) if mock_patients[i].conditions else "None"
            table_data.append([p_id, conds, f"{decision} {icon}", reason])
            
        return t_info, eval_md, table_data
    except Exception as e:
        return f"Error", f"Error: {e}", []

with gr.Blocks(title="Clinical Trial OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Clinical Trial AI Coordination Suite")
    
    with gr.Tabs():
        with gr.TabItem("🩺 1. Patient Diagnostic Assistant"):
            gr.Markdown("### Diagnostic View (1 Patient vs Regional Trials)")
            with gr.Row():
                with gr.Column(scale=1):
                    diag_btn = gr.Button("🔍 Run Diagnostic Scan", variant="primary")
                    diag_p_info = gr.Markdown("Loading...")
                with gr.Column(scale=2):
                    diag_eval = gr.Markdown("...")
            diag_table = gr.DataFrame(headers=["Trial ID", "Required Conditions", "AI Decision", "Coordinator Reasoning"], wrap=True)
            
            diag_btn.click(run_diagnostic, outputs=[diag_p_info, diag_eval, diag_table])
            
        with gr.TabItem("📊 2. Batch Recruitment System"):
            gr.Markdown("### Recruitment View (1 Trial vs Patient Database)")
            with gr.Row():
                with gr.Column(scale=1):
                    rec_btn = gr.Button("📈 Run Batch Scan", variant="primary")
                    rec_t_info = gr.Markdown("Loading...")
                with gr.Column(scale=2):
                    rec_eval = gr.Markdown("...")
            rec_table = gr.DataFrame(headers=["Patient Name", "Conditions", "AI Decision", "Coordinator Reasoning"], wrap=True)
            
            rec_btn.click(run_recruitment, outputs=[rec_t_info, rec_eval, rec_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
