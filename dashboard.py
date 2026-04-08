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

def build_diagnostic_prompt(obs):
    patient = obs.patient.model_dump()
    trials = [t.model_dump() for t in obs.trials]
    return f"""
You are an expert Clinical Trial Coordinator. 
DIAGNOSTIC MODE: Evaluate ONE patient against ALL available trials.

PATIENT: {json.dumps(patient)}
TRIALS: {json.dumps(trials)}

Output JSON: {{
  "reasoning_summary": "...",
  "trial_evaluations": [...],
  "ranked_trial_ids": [...],
  "confidence": 0.9
}}
"""

def build_recruitment_prompt(trial, patients):
    return f"""
You are an expert Recruitment Specialist. 
RECRUITMENT MODE: Evaluate ONE trial against a batch of patients.

TRIAL: {json.dumps(trial.model_dump())}
PATIENTS: {json.dumps([p.model_dump() for p in patients])}

Output JSON: {{
  "reasoning_summary": "...",
  "trial_evaluations": [...],
  "ranked_trial_ids": [...],
  "confidence": 0.9
}}
"""

def run_diagnostic():
    obs = env.reset()
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    prompt = build_diagnostic_prompt(obs)
    
    response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    result = json.loads(response.choices[0].message.content)
    action = Action(**result)
    _, reward, _, _ = env.step(action)

    p_info = f"**Patient:** {obs.task_id} ({obs.patient.age}y, {obs.patient.gender})\n**History:** {', '.join(obs.patient.conditions)}"
    eval_md = f"### AI Reasoning\n{action.reasoning_summary}\n\n**Match Grade:** {reward*100:.1f}%"
    
    table_data = []
    for te in action.trial_evaluations:
        table_data.append([te.trial_id, te.decision.upper()])
        
    return p_info, eval_md, table_data

def run_recruitment():
    obs = env.reset() # Get a random case to pick a trial and patients
    trial = obs.trials[0]
    # In a real batch, we'd have multiple patients. For demo, we'll use 3 versions of the patient or siblings
    mock_patients = [obs.patient] 
    
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    prompt = build_recruitment_prompt(trial, mock_patients)
    
    response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    result = json.loads(response.choices[0].message.content)
    action = Action(**result)
    
    t_info = f"**Trial ID:** {trial.id}\n**Criteria:** {', '.join(trial.required_conditions)}"
    eval_md = f"### Recruitment Summary\n{action.reasoning_summary}"
    
    table_data = []
    for te in action.trial_evaluations:
        table_data.append([obs.task_id, te.decision.upper()])
        
    return t_info, eval_md, table_data

with gr.Blocks(title="Clinical Trial OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Clinical Trial AI Coordination Suite")
    
    with gr.Tabs():
        with gr.TabItem("🩺 1. Patient Diagnostic Assistant"):
            gr.Markdown("### One Patient vs. Regional Trial Catalog")
            diag_btn = gr.Button("🔍 Run Patient Analysis", variant="primary")
            with gr.Row():
                diag_p_info = gr.Markdown("Select patient to see details...")
                diag_eval = gr.Markdown("AI analysis will appear here...")
            diag_table = gr.DataFrame(headers=["Trial ID", "Match Status"])
            
            diag_btn.click(run_diagnostic, outputs=[diag_p_info, diag_eval, diag_table])
            
        with gr.TabItem("📊 2. Batch Recruitment System"):
            gr.Markdown("### One Trial vs. Hospital Patient Database")
            rec_btn = gr.Button("📈 Run Batch Recruitment", variant="primary")
            with gr.Row():
                rec_t_info = gr.Markdown("Select trial to scan database...")
                rec_eval = gr.Markdown("Recruitment stats will appear here...")
            rec_table = gr.DataFrame(headers=["Patient ID", "Eligibility"])
            
            rec_btn.click(run_recruitment, outputs=[rec_t_info, rec_eval, rec_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
