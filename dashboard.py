import os
import json
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from environment import ClinicalTrialEnv
from models import Action

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
  "reasoning_summary": "Write 2 sentences explaining your medical thought process.",
  "trial_evaluations": [
    {
      "trial_id": "string",
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
  "ranked_trial_ids": ["trial_id_1", "trial_id_2"],
  "confidence": 0.0 to 1.0
}
"""

def build_diagnostic_prompt(obs):
    patient = obs.patient.model_dump()
    trials = [t.model_dump() for t in obs.trials]
    return f"""
You are an expert Clinical Trial Coordinator AI. 
DIAGNOSTIC MODE: Evaluate ONE patient against ALL available trials.

PATIENT RECORD:
{json.dumps(patient, indent=2)}

AVAILABLE TRIALS:
{json.dumps(trials, indent=2)}

{get_shared_schema()}
"""

def build_recruitment_prompt(trial, patients):
    return f"""
You are an expert Recruitment Specialist. 
RECRUITMENT MODE: Evaluate ONE trial against a batch of patients.

TRIAL CRITERIA:
{json.dumps(trial.model_dump(), indent=2)}

PATIENT DATABASE:
{json.dumps([p.model_dump() for p in patients], indent=2)}

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
        for te in action.trial_evaluations:
            icon = "✅" if te.decision == "eligible" else ("❌" if te.decision == "ineligible" else "⚠️")
            table_data.append([te.trial_id, te.decision.upper(), icon])
            
        return p_info, eval_md, table_data
    except Exception as e:
        return f"Error loading patient", f"AI Processing Error: {e}", []

def run_recruitment():
    obs = env.reset() # Get random case
    trial = obs.trials[0]
    # Small batch for demo
    mock_patients = [obs.patient] 
    
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    prompt = build_recruitment_prompt(trial, mock_patients)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        action = Action(**result)
        
        t_info = f"**Trial ID:** {trial.id}\n**Required Conditions:** {', '.join(trial.required_conditions)}"
        eval_md = f"### 📊 Recruitment Analysis\n> {action.reasoning_summary}"
        
        table_data = []
        for te in action.trial_evaluations:
            icon = "✅" if te.decision == "eligible" else ("❌" if te.decision == "ineligible" else "⚠️")
            table_data.append([obs.task_id, te.decision.upper(), icon])
            
        return t_info, eval_md, table_data
    except Exception as e:
        return f"Error loading trial", f"AI Processing Error: {e}", []

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
