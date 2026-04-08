import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.table import Table

from environment import ClinicalTrialEnv
from models import Action

load_dotenv()  # Load all keys from .env automatically

console = Console()

# --- CONFIGURATION (Hackathon Checklist Compliant) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Use HF_TOKEN if provided by the OpenEnv grader, otherwise fallback to OPENAI_API_KEY from .env
API_KEY = HF_TOKEN if HF_TOKEN else os.getenv("OPENAI_API_KEY", "")

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

def run_agent():
    console.print(Panel.fit("[bold blue]🏥 Clinical Trial OpenEnv - AI Coordinator Benchmark[/bold blue]"))
    
    env = ClinicalTrialEnv()
    
    # We MUST run at least 3 tasks to pass Phase 2 validation
    for i in range(3):
        console.print(f"\n[bold green]🏁 Starting Benchmark Task {i+1}/3[/bold green]")
        
        # 1. RESET
        obs = env.reset()
        # REQUIRED BY CHECKLIST: [START] task=ID
        print(f"[START] task={obs.task_id}", flush=True)

        console.print(f"  [bold green]▶ Case ID:[/bold green] {obs.task_id}")
        console.print(f"  [yellow]- Patient Profile:[/yellow] Age {obs.patient.age}, {obs.patient.gender}")
        conds = ", ".join(obs.patient.conditions) if obs.patient.conditions else "None"
        console.print(f"  [yellow]- Medical Conditions:[/yellow] {conds}")
        console.print(f"  [yellow]- Clinical Trials Available:[/yellow] {len(obs.trials)}")
        
        # 2. STATE (Logic)
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        prompt = build_prompt(obs)
        
        console.print("  [cyan]🤖 Injecting State Observation into Llama Coordinator...[/cyan]")
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a specialized JSON parser agent assisting with clinical trials. Do not wrap JSON in markdown blocks."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            result_str = response.choices[0].message.content
            result_json = json.loads(result_str)
            action = Action(**result_json)
            
        except Exception as e:
            console.print(f"\n[bold red]❌ LLM Failure. Could not parse action correctly. Error: {e}[/bold red]")
            # Log failure but continue to next task
            print(f"[END] task={obs.task_id} score=0.01 steps=1", flush=True)
            continue
            
        # 3. STEP
        next_obs, reward, done, info = env.step(action)
        # REQUIRED BY CHECKLIST: [STEP] step=IDX reward=VAL
        print(f"[STEP] step=1 reward={reward}", flush=True)
            
        console.print(f"  [cyan]⚖️ Grade Received: {reward*100:.1f}%[/cyan]")
        
        # --- PRO-HACKATHON PRESENTATION ---
        console.print(Panel(f"[italic cyan]\"{action.reasoning_summary}\"[/italic cyan]", title=f"🧠 Reasoning for {obs.task_id}", border_style="cyan"))
        
        # 4. END
        # REQUIRED BY CHECKLIST: [END] task=ID score=VAL steps=N
        print(f"[END] task={obs.task_id} score={reward} steps=1", flush=True)

    console.print("\n[dim italic]Multi-task benchmark complete. Total 3 scenarios evaluated.[/dim italic]\n")

if __name__ == "__main__":
    run_agent()
