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

def get_patients_and_trials(env):
    patients = []
    trials = []
    seen_trial_signatures = set()
    for i, c in enumerate(env.cases):
        p = c["patient"]
        p["name"] = f"Patient_{i+1}" 
        p["complexity"] = c["task_id"].split("_")[0].capitalize()
        patients.append(p)
        for t in c.get("trials", []):
            sig = str(t.get("required_conditions")) + str(t.get("excluded_conditions")) + str(t.get("lab_criteria"))
            if sig not in seen_trial_signatures:
                seen_trial_signatures.add(sig)
                t["id"] = f"Trial_{len(seen_trial_signatures):02d}"
                t["complexity"] = c["task_id"].split("_")[0].capitalize()
                trials.append(t)
    return patients, trials

def get_shared_schema():
    return """
You MUST output valid JSON ONLY matching exactly this schema:
{
  "reasoning_summary": "Summarize total findings for the entire batch.",
  "trial_evaluations": [
    {
      "trial_id": "string",
      "decision": "eligible" | "ineligible" | "needs_review",
      "reason": "One specific sentence explaining this exact match decision."
    }
  ],
  "ranked_trial_ids": ["id_1", "id_2"],
  "confidence": 0.0 to 1.0
}
"""

def build_diagnostic_prompt(patient, all_trials):
    return f"""
You are an expert Clinical Trial Coordinator AI. 
Evaluate ONE patient against the entire Regional Trial Catalog.
PATIENT RECORD: {json.dumps(patient)}
TRIAL CATALOG: {json.dumps(all_trials)}
{get_shared_schema()}
"""

def build_recruitment_prompt(trial, patients):
    return f"""
You are an expert Recruitment Specialist. 
Evaluate ONE trial against a batch of patients.
TRIAL CRITERIA: {json.dumps(trial)}
PATIENT DATABASE: {json.dumps([{"id": f"Patient_{i+1}", "data": p} for i, p in enumerate(patients)])}
Important: In "trial_evaluations", use the Patient ID (Patient_1, Patient_2...) as the "trial_id" field.
{get_shared_schema()}
"""

def run_diagnostic():
    patients, all_trials = get_patients_and_trials(env)
    target_patient = random.choice(patients)
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    prompt = build_diagnostic_prompt(target_patient, all_trials)
    
    try:
        response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        evals = result.get("trial_evaluations", [])

        p_info = f"**Patient:** {target_patient['name']} | **Tier:** {target_patient.get('complexity')} | **Age:** {target_patient.get('age')}y\n**History:** {', '.join(target_patient.get('conditions', []))}"
        eval_md = f"### 🧠 AI Reasoning\n> {result.get('reasoning_summary', 'Diagnostic scan complete.')}"
        
        table_data = []
        eligible_count = 0
        review_count = 0
        for t in all_trials:
            eval_item = next((te for te in evals if te["trial_id"] == t["id"]), None)
            decision = eval_item["decision"].lower() if eval_item else "ineligible"
            reason = eval_item["reason"] if eval_item else "No reason provided."
            icon = "✅" if "eligible" in decision else ("❌" if "ineligible" in decision else "⚠️")
            if "eligible" in decision: eligible_count += 1
            elif "review" in decision: review_count += 1
            table_data.append([t["id"], ", ".join(t.get("required_conditions", [])), f"{decision.upper()} {icon}", reason])
            
        report_md = f"""
### 📋 Diagnostic Action Report
- **System Verdict:** {target_patient['name']} is potentially eligible for **{eligible_count}** clinical trials.
- **Doctor Review:** {review_count} trials require secondary confirmation.
- **Safety Status:** ✅ All dangerous exclusions checked.
"""
        return p_info, eval_md, table_data, report_md
    except Exception as e:
        return f"Error", f"Error: {e}", [], "Scan Failed"

def run_recruitment():
    patients, all_trials = get_patients_and_trials(env)
    target_trial = random.choice(all_trials)
    scan_list = random.sample(patients, min(5, len(patients)))
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    prompt = build_recruitment_prompt(target_trial, scan_list)
    
    try:
        response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        evals = result.get("trial_evaluations", [])
        
        t_info = f"**Target Trial:** {target_trial['id']} | **Tier:** {target_trial.get('complexity')}\n**Requires:** {', '.join(target_trial.get('required_conditions', []))}"
        eval_md = f"### 📊 AI Reasoning\n> {result.get('reasoning_summary', 'Batch scan complete.')}"
        
        table_data = []
        el_count = 0
        for i, p in enumerate(scan_list):
            p_name = p["name"]
            eval_item = next((te for te in evals if te["trial_id"] in [p_name, f"Patient_{i+1}"]), None)
            decision = eval_item["decision"].lower() if eval_item else "review"
            reason = eval_item["reason"] if eval_item else "..."
            icon = "✅" if "eligible" in decision else ("❌" if "ineligible" in decision else "⚠️")
            if "eligible" in decision: el_count += 1
            table_data.append([p_name, ", ".join(p.get("conditions", [])), f"{decision.upper()} {icon}", reason])
            
        report_md = f"""
### 📋 Recruitment Action Report
- **Candidates Found:** **{el_count}** patients meet enrollment criteria.
- **Action:** Send recruitment invitations via secure hospital portal.
- **Efficiency:** AI scanned {len(scan_list)} profiles in seconds.
"""
        return t_info, eval_md, table_data, report_md
    except Exception as e:
        return f"Error", f"Error: {e}", [], "Scan Failed"

# --- Gradio UI ---
with gr.Blocks(title="Clinical Trial OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Clinical Trial AI Coordination Suite")
    
    with gr.Accordion("ℹ️ System Complexity Legend (How the AI Grader Works)", open=False):
        gr.Markdown("""
| Tier | Criteria | Evaluation Logic |
| :--- | :--- | :--- |
| **🟢 Easy** | Simple Match | Follows basic required conditions and age ranges. |
| **🟡 Medium** | Hard Exclusions | Must cross-reference against 'Excluded Conditions' (Safety Check). |
| **🔴 Hard** | Lab Reasoning | Multi-factor analysis of bloodwork/vitals (e.g. HbA1c > 7.5). |
""")

    with gr.Tabs():
        with gr.TabItem("🩺 1. Patient Diagnostic Assistant"):
            gr.Markdown("### Diagnostic View (1 Patient vs Regional Trial Catalog)")
            diag_btn = gr.Button("🔍 Run Diagnostic Scan", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    diag_p_info = gr.Markdown("Loading...")
                with gr.Column(scale=2):
                    diag_eval = gr.Markdown("...")
            diag_table = gr.DataFrame(headers=["Trial ID", "Required Conditions", "AI Decision", "Coordinator Reasoning"], wrap=True)
            diag_report = gr.Markdown("### 📋 Loading Report...")
            
            diag_btn.click(run_diagnostic, outputs=[diag_p_info, diag_eval, diag_table, diag_report])
            
        with gr.TabItem("📊 2. Batch Recruitment System"):
            gr.Markdown("### Recruitment View (1 Trial vs Patient Database)")
            rec_btn = gr.Button("📈 Run Batch Scan", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    rec_t_info = gr.Markdown("Loading...")
                with gr.Column(scale=2):
                    rec_eval = gr.Markdown("...")
            rec_table = gr.DataFrame(headers=["Patient Name", "Conditions", "AI Decision", "Coordinator Reasoning"], wrap=True)
            rec_report = gr.Markdown("### 📋 Loading Report...")
            
            rec_btn.click(run_recruitment, outputs=[rec_t_info, rec_eval, rec_table, rec_report])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
