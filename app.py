import os
from fastapi import FastAPI, Body
from environment import ClinicalTrialEnv
from models import Action
from typing import Dict, Any

app = FastAPI(title="Clinical Trial OpenEnv API")

# Initialize the environment
env = ClinicalTrialEnv()

@app.get("/")
def health_check():
    return {"status": "running", "environment": "clinical-trial-openenv"}

@app.post("/reset")
def reset():
    """Reset the environment and return the initial observation."""
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action_data: Dict[str, Any] = Body(...)):
    """Take a step in the environment using the provided action."""
    # Convert dict action to Pydantic Action model
    action = Action(**action_data)
    next_obs, reward, done, info = env.step(action)
    
    return {
        "observation": next_obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    """Return the current state of the environment."""
    return env.state().model_dump()

def start_server():
    import uvicorn
    # Hugging Face Spaces port is usually 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    start_server()
