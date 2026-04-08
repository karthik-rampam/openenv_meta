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
  "reasoning_summary": "Summarize the matching logic for the entire batch.",
  "trial_evaluations": [
    {
      "trial_id": "string (the ID of the trial OR the ID of the patient being evaluated)",
      "decision": "eligible" | "ineligible" | "needs_review",
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
DIAGNOSTIC MODE: Evaluate ONE patient against ALL provided trials.
You MUST provide a separate evaluation for EACH trial ID in the list.

PATIENT RECORD:
{json.dumps(patient, indent=2)}

AVAILABLE TRIALS:
{json.dumps(trials, indent=2)}

{get_shared_schema()}
"""

def build_recruitment_prompt(trial, patients):
    # For recruitment mode, we treat trial_id in the schema as the Patient ID
    return f"""
You are an expert Recruitment Specialist. 
RECRUITMENT MODE: Evaluate ONE trial against MULTIPLE patients.
You MUST provide a separate evaluation for EACH Patient ID in the list.

TRIAL CRITERIA:
{json.dumps(trial.model_dump(), indent=2)}

PATIENT DATABASE (BATCH):
{json.dumps([{"id": f"P{i+1}", "data": p.model_dump()} for i, p in enumerate(patients)], indent=2)}

Special Rule: In the "trial_evaluations" list, use the Patient ID (P1, P2, etc.) as the "trial_id" field.

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

        p_info = f"**Patient:** {obs.task_id} ({obs.patient.age}y, {obs.patient.gender})\n**History:** {', '.join(obs.patient.conditions)}"
        eval_md = f"### 🧠 AI Reasoning\n> {action.reasoning_summary}\n\n**⚖️ Match Grade:** {reward*100:.1f}%"
        
        table_data = []
        # Ensure we show all trials by iterating over the environmental trials
        for t in obs.trials:
            eval_item = next((te for te in action.trial_evaluations if te.trial_id == t.id), None)
            decision = eval_item.decision.upper() if eval_item else "PENDING"
            icon = "✅" if decision == "ELIGIBLE" else ("❌" if decision == "INELIGIBLE" else "⚠️")
            table_data.append([t.id, decision, icon])
            
        return p_info, eval_md, table_data
    except Exception as e:
        return f"Error loading patient", f"AI Processing Error: {e}", []

def run_recruitment():
    # Pick a random case to get a trial
    obs = env.reset()
    trial = obs.trials[0]
    
    # Sample 5 random patients from the entire case list for a real "batch" feel
    raw_cases = env.cases
    sample_size = min(5, len(raw_cases))
    random_cases = random.sample(raw_cases, sample_size)
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
        
        # We parse manually as Action expects trial_ids, but here IDs are patients
        summary = result.get("reasoning_summary", "Batch scan complete.")
        evals = result.get("trial_evaluations", [])
        
        t_info = f"**Trial ID:** {trial.id}\n**Target Conditions:** {', '.join(trial.required_conditions)}"
        eval_md = f"### 📊 Recruitment Analysis\n> {summary}"
        
        table_data = []
        for i, p_id in enumerate(patient_ids):
            # Try to find evaluation by P1, P2... or by the patient ID itself
            eval_item = next((te for te in evals if te["trial_id"] in [f"P{i+1}", p_id]), None)
            decision = eval_item["decision"].upper() if eval_item else "NEEDS_REVIEW"
            icon = "✅" if decision == "ELIGIBLE" else ("❌" if decision == "INELIGIBLE" else "⚠️")
            table_data.append([p_id, decision, icon])
            
        return t_info, eval_md, table_data
    except Exception as e:
        return f"Error loading trial", f"AI Processing Error: {e}", []

# --- Gradio UI ---
with gr.Blocks(title="Clinical Trial OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Clinical Trial AI Coordination Suite")
    
    with gr.Tabs():
        with gr.TabItem("🩺 1. Patient Diagnostic Assistant"):
            gr.Markdown("### One Patient vs. Regional Trial Catalog")
            with gr.Row():
                with gr.Column(scale=1):
                    diag_btn = gr.Button("🔍 Run Patient Analysis", variant="primary")
                    diag_p_info = gr.Markdown("Click button to load scenario...")
                with gr.Column(scale=2):
                    diag_eval = gr.Markdown("AI analysis results...")
                    diag_table = gr.DataFrame(headers=["Trial ID", "Match Status", "Icon"], datatype=["str", "str", "str"])
            
            diag_btn.click(run_diagnostic, outputs=[diag_p_info, diag_eval, diag_table])
            
        with gr.TabItem("📊 2. Batch Recruitment System"):
            gr.Markdown("### One Trial vs. Hospital Patient Database")
            with gr.Row():
                with gr.Column(scale=1):
                    rec_btn = gr.Button("📈 Run Batch Recruitment", variant="primary")
                    rec_t_info = gr.Markdown("Click button to load recruitment case...")
                with gr.Column(scale=2):
                    rec_eval = gr.Markdown("Recruitment summary...")
                    rec_table = gr.DataFrame(headers=["Patient ID", "Eligibility", "Icon"], datatype=["str", "str", "str"])
            
            rec_btn.click(run_recruitment, outputs=[rec_t_info, rec_eval, rec_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
