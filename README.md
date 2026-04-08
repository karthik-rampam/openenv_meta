# AI Clinical Trial Coordinator 🏥

A Meta Llama 3 powered healthcare platform that automates clinical trial matching. We built two distinct AI products to solve the biggest bottlenecks in medical research recruitment:

## 1. Pharma Batch Recruitment System (`run_batch_recruitment.py`)
Pharmaceutical companies and research hospitals waste months reading patient files to find eligible volunteers. This script takes **1 specific clinical trial** and instantly scans an entire hospital database of patients, outputting a precise, automated call-list of eligible candidates while filtering out dangerous medical exclusions.

```bash
python run_batch_recruitment.py
```

## 2. Patient Diagnostic Assistant (`run_patient_analysis.py`)
Doctors don't have time to cross-reference a walk-in patient against 50 different experimental treatments. This script takes **1 specific Patient** and evaluates them simultaneously against the entire regional catalog of clinical trials, providing a beautiful dashboard showing exactly what life-saving studies they qualify for.

```bash
python run_patient_analysis.py
```

## Architecture
- Powered by **Meta Llama 3.3 (70B)** via Groq for blazing-fast medical reasoning.
- Uses a Strict Validation Engine (`models.py` & `environment.py`) that formats unstructured medical files securely into strict JSON schemas before the LLM touches them, preventing hallucinations.
- Features gorgeous, judge-friendly terminal dashboards using the `Rich` Python library.

## Setup & Installation
```bash
pip install -r requirements.txt
```
*Note: Ensure your Groq API Key is pasted into the Configuration section of the python scripts before running!*
