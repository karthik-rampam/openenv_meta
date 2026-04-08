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

def build_prompt(obs):
    patient = obs.patient.model_dump()
    trials = [t.model_dump() for t in obs.trials]
    
    prompt = f"""
You are an expert Clinical Trial Coordinator AI. 
Evaluate the following patient against the provided clinical trials.

PATIENT RECORD:
{json.dumps(patient, indent=2)}

AVAILABLE TRIALS:
{json.dumps(trials, indent=2)}

YOUR TASK:
Determine if the patient is "eligible", "ineligible", or "needs_review" for EACH trial.
Also, rank the trials from most appropriate to least appropriate.

You MUST output valid JSON ONLY matching exactly this schema:
{{
  "reasoning_summary": "Write 2 sentences explaining your medical thought process.",
  "trial_evaluations": [
    {{
      "trial_id": "string",
      "decision": "eligible" | "ineligible" | "needs_review",
      "criterion_evaluations": [
        {{
          "criterion_name": "string",
          "met": true/false,
          "reason": "string"
        }}
      ]
    }}
  ],
  "ranked_trial_ids": ["trial_id_1", "trial_id_2"],
  "confidence": 0.0 to 1.0
}}
"""
    return prompt

def run_analysis():
    obs = env.reset()
    
    # 1. Prepare Info for Display
    patient_md = f"**ID:** {obs.task_id} | **Age:** {obs.patient.age} | **Gender:** {obs.patient.gender}\n\n**Conditions:** {', '.join(obs.patient.conditions) if obs.patient.conditions else 'None'}"
    
    trial_md = ""
    for t in obs.trials:
        trial_md += f"- **{t.id}**: Requires ({', '.join(t.required_conditions)})\n"

    # 2. AI Call
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    prompt = build_prompt(obs)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        result_json = json.loads(response.choices[0].message.content)
        action = Action(**result_json)
        
        # 3. Grade
        next_obs, reward, done, info = env.step(action)
        
        # 4. Format Output
        eval_md = f"### 🧠 AI Reasoning\n> {action.reasoning_summary}\n\n"
        eval_md += f"### ⚖️ Grader Score: **{reward*100:.1f}%**\n\n"
        
        results_data = []
        for t_eval in action.trial_evaluations:
            icon = "✅" if t_eval.decision == "eligible" else ("❌" if t_eval.decision == "ineligible" else "⚠️")
            results_data.append([t_eval.trial_id, t_eval.decision.upper(), icon])
            
        return patient_md, trial_md, eval_md, results_data
        
    except Exception as e:
        return patient_md, trial_md, f"Error: {e}", []

# --- Gradio UI ---
with gr.Blocks(title="Clinical Trial AI Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Clinical Trial AI Coordinator Dashboard")
    gr.Markdown("Click the button below to load a random patient scenario and see the Llama 3 AI process their eligibility.")
    
    with gr.Row():
        with gr.Column(scale=1):
            btn = gr.Button("🔄 Run New Analysis", variant="primary")
            gr.Markdown("### 👤 Patient Information")
            patient_info = gr.Markdown("Click button to load...")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🧪 Available Trials")
            trials_info = gr.Markdown("...")

    with gr.Row():
        with gr.Column(scale=2):
            analysis_output = gr.Markdown("### 🤖 AI Evaluation Result")
            
        with gr.Column(scale=1):
            results_table = gr.HighlightedText(label="Match Results")
            results_df = gr.DataFrame(headers=["Trial ID", "Decision", "Status"], datatype=["str", "str", "str"])

    btn.click(
        fn=run_analysis,
        outputs=[patient_info, trials_info, analysis_output, results_df]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
