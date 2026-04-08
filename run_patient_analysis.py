import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from environment import ClinicalTrialEnv

load_dotenv()

console = Console()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")

def get_patients_and_trials(env):
    patients = []
    trials = []
    seen_trial_signatures = set()
    
    for i, c in enumerate(env.cases):
        p = c["patient"]
        p["name"] = f"Patient_{i+1}" 
        patients.append(p)
        
        for t in c.get("trials", []):
            sig = str(t.get("required_conditions")) + str(t.get("excluded_conditions")) + str(t.get("lab_criteria"))
            if sig not in seen_trial_signatures:
                seen_trial_signatures.add(sig)
                t["id"] = f"Trial_{len(seen_trial_signatures):02d}"
                trials.append(t)
                
    return patients, trials

def run_analysis():
    console.print(Panel.fit("[bold blue]🏥 Hospital Patient Trial-Matching System[/bold blue]"))
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    console.print("\n[bold magenta]🔄 [OpenEnv Diagnostic: Loading Regional Catalog][/bold magenta]")
    env = ClinicalTrialEnv()
    patients, all_trials = get_patients_and_trials(env)
    
    target_patient = random.choice(patients)
    
    console.print("\n[bold magenta]👁️  [OpenEnv Diagnostic: Extracting Patient State Observation][/bold magenta]")
    console.print(f"  [bold green]👨‍⚕️ Target Patient Selected:[/bold green] {target_patient['name']}")
    console.print(f"  [yellow]- Age:[/yellow] {target_patient.get('age')}, {target_patient.get('gender')}")
    console.print(f"  [yellow]- Conditions:[/yellow] {', '.join(target_patient.get('conditions', [])) if target_patient.get('conditions') else 'None'}")
    console.print(f"  [yellow]- Scanning Catalog:[/yellow] Evaluating against {len(all_trials)} distinct Clinical Trials...\n")

    prompt = f"""
You are a Clinical Trial Coordinator. 
You have ONE patient. You must cross-reference them against this catalog of clinical trials.
PATIENT: {json.dumps(target_patient)}

TRIAL CATALOG: {json.dumps(all_trials)}

Return strictly valid JSON ONLY combining these rules:
{{
  "trial_evaluations": [
    {{
      "trial_id": "string (the exact Trial_XX ID)",
      "decision": "eligible" | "ineligible" | "needs_review",
      "reason": "1 short sentence explaining why"
    }}
  ]
}}
Ensure EVERY trial in the catalog has exactly one evaluation entry. Do not hallucinate trial IDs.
"""

    console.print("[bold magenta]🕹️  [OpenEnv Diagnostic: Executing Catalog Step Analysis][/bold magenta]")
    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}")) as progress:
        task_id = progress.add_task("AI Coordinator running full cross-diagnostic against all hospital trials...", total=1)
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a JSON parser."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            res = json.loads(response.choices[0].message.content)
            evals = res.get("trial_evaluations", [])
            progress.update(task_id, advance=1)
            
        except Exception as e:
            console.print(f"[bold red]LLM Error: {e}[/bold red]")
            return
            
    # --- DISPLAY TABLE ---
    console.print("\n[bold magenta]📊 COMPREHENSIVE PATIENT MATCHING RESULTS[/bold magenta]")
    
    table = Table(show_header=True, header_style="bold white")
    table.add_column("Trial ID", style="cyan")
    table.add_column("AI Decision", justify="center")
    table.add_column("Coordinator Reasoning", style="italic")
    
    eligible_count = 0
    review_count = 0
    
    def sort_key(x):
        d = x.get("decision", "ineligible")
        if d == "eligible": return 0
        if d == "needs_review": return 1
        return 2
        
    evals.sort(key=sort_key)
    
    for eval in evals:
        decision = eval.get("decision", "ineligible")
        tid = eval.get("trial_id", "?")
        reason = eval.get("reason", "")
        
        if decision == "eligible":
            eligible_count += 1
            table.add_row(tid, "[bold green]ELIGIBLE ✅[/bold green]", reason)
        elif decision == "needs_review":
            review_count += 1
            table.add_row(tid, "[bold yellow]REVIEW ⚠️[/bold yellow]", reason)
        else:
            table.add_row(tid, "[bold red]INELIGIBLE ❌[/bold red]", reason)
        
    console.print(table)
    
    cond_str = ", ".join(target_patient.get("conditions", [])) if target_patient.get("conditions") else "None"
    summary = f"""
[bold cyan]Patient Profile:[/bold cyan] {target_patient.get('age')} year-old {target_patient.get('gender')}
[bold cyan]Conditions:[/bold cyan] {cond_str}
[bold cyan]Labs recorded:[/bold cyan] {json.dumps(target_patient.get('lab_results', {}))}

[bold green]System Verdict:[/bold green] {target_patient['name']} is definitively eligible for [bold white]{eligible_count}[/bold white] clinical trials!
[dim yellow]({review_count} trials require a secondary doctor review).[/dim yellow]
"""
    console.print(Panel(summary.strip(), title="📋 Diagnostic Action Report", border_style="green"))

if __name__ == "__main__":
    run_analysis()
