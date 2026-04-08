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
    for i, c in enumerate(env.cases):
        p = c["patient"]
        p["name"] = f"Patient_{i+1}" 
        patients.append(p)
        for t in c.get("trials", []):
            if t not in trials:
                trials.append(t)
    return patients, trials

def run_batch():
    console.print(Panel.fit("[bold blue]🏥 Pharma Trial Batch Recruitment System[/bold blue]"))
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    console.print("\n[bold magenta]🔄 [OpenEnv Batch: Loading Hospital Database][/bold magenta]")
    env = ClinicalTrialEnv()
    patients, all_trials = get_patients_and_trials(env)
    
    target_trial = random.choice(all_trials)
    
    console.print("\n[bold magenta]👁️  [OpenEnv Batch: Extracting Target Trial State][/bold magenta]")
    console.print(f"  [bold green]🎯 Target Pharma Trial Picked:[/bold green] {target_trial['id']}")
    console.print(f"  [yellow]- Required:[/yellow] {target_trial.get('required_conditions', [])}")
    console.print(f"  [yellow]- Excluded:[/yellow] {target_trial.get('excluded_conditions', [])}")
    
    scan_list = random.sample(patients, min(8, len(patients)))
    console.print(f"  [yellow]- Scanning Database:[/yellow] Evaluating {len(scan_list)} Patients...\n")

    results = {"eligible": [], "needs_review": [], "ineligible": []}

    console.print("[bold magenta]🕹️  [OpenEnv Batch: Executing Multi-Agent Step][/bold magenta]")
    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}")) as progress:
        task_id = progress.add_task("AI Coordinator analyzing massive patient sets...", total=len(scan_list))
        
        for patient in scan_list:
            prompt = f"""
You are a Clinical Trial Coordinator. 
Does this patient qualify for the trial?
PATIENT: {json.dumps(patient)}
TRIAL: {json.dumps(target_trial)}

Return strictly valid JSON ONLY combining these rules:
{{
  "decision": "eligible" | "ineligible" | "needs_review",
  "reason": "1 short sentence explaining why"
}}
"""
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
                decision = res.get("decision", "ineligible")
                reason = res.get("reason", "No reason provided.")
                
                if decision not in results:
                    decision = "ineligible"
                    
                results[decision].append((patient["name"], patient.get("conditions", []), reason))
            except Exception as e:
                results["ineligible"].append((patient["name"], [], f"API Error: {e}"))
                
            progress.update(task_id, advance=1)
            
    # --- DISPLAY TABLE ---
    console.print("\n[bold magenta]📊 BATCH RECRUITMENT RESULTS[/bold magenta]")
    
    table = Table(show_header=True, header_style="bold white")
    table.add_column("Patient Name", style="cyan")
    table.add_column("Conditions", style="dim")
    table.add_column("AI Decision", justify="center")
    table.add_column("Coordinator Reasoning", style="italic")
    
    for (name, conds, reason) in results["eligible"]:
        table.add_row(name, ", ".join(conds) if conds else "None", "[bold green]ELIGIBLE ✅[/bold green]", reason)
    for (name, conds, reason) in results["needs_review"]:
        table.add_row(name, ", ".join(conds) if conds else "None", "[bold yellow]REVIEW ⚠️[/bold yellow]", reason)
    for (name, conds, reason) in results["ineligible"]:
        table.add_row(name, ", ".join(conds) if conds else "None", "[bold red]REJECTED ❌[/bold red]", reason)
        
    console.print(table)
    
    summary = f"""
[bold green]Eligible Call List:[/bold green] {len(results['eligible'])} patients
[bold yellow]Needs Doctor Review:[/bold yellow] {len(results['needs_review'])} patients
[bold red]Ineligible/Rejected:[/bold red] {len(results['ineligible'])} patients
"""
    console.print(Panel(summary.strip(), title="📋 Final Action Report", border_style="green"))

if __name__ == "__main__":
    run_batch()
