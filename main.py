import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI  # Import OpenAI to talk to LM Studio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LM Studio Configuration ---
# Ensure LM Studio Local Server is running on port 1234
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

class TrafficRequest(BaseModel):
    traffic_level: str 
    vehicle_type: str  
    objective: str     
    source: str
    destination: str

ROUTE_OPTIONS = {
    "d-h": [
        {"name": "Route 1 (Highway)", "path": ["d", "f", "k", "a", "b", "c", "o", "h"], "base_speed": 28.5},
        {"name": "Route 2 (City Detour)", "path": ["d", "f", "e", "g", "j", "h"], "base_speed": 18.5},
        {"name": "Route 3 (Side Streets)", "path": ["d", "f", "m", "b", "c", "o" ,"h"], "base_speed": 10.09}
    ],
    "f-h": [
        {"name": "Route 1", "path": ["f", "e", "i", "j", "h"], "base_speed": 22.5 },
        {"name": "Route 2", "path": ["f", "e", "g", "n", "b", "c", "p", "h"], "base_speed": 7.5},
        {"name": "Route 3", "path": ["f", "m", "b", "c", "o", "h"], "base_speed": 14.0 }
    ],
    "d-f": [{"name": "Direct", "path": ["d", "f"], "base_speed": 30}],
    "e-h": [
        {"name": "Route 1", "path": ["e", "i", "j", "h"], "base_speed": 30.0},
        {"name": "Route 2", "path": ["e", "g", "n", "b", "c", "o", "h"], "base_speed": 20.0 },
        {"name": "Route 3", "path": ["e", "f", "k", "a", "b", "c" ,"p" ,"h"], "base_speed": 15.5}
    ],
    "e-f": [
        {"name": "Route 1", "path": ["e", "f"], "base_speed": 27.8},
        {"name": "Route 2", "path": ["e", "g", "l", "a", "k", "f"], "base_speed": 16.2},
        {"name": "Route 3", "path": ["e", "g", "n", "b", "a", "k" ,"f"], "base_speed": 9.4}
    ],
    "h-f": [
        {"name": "Route 1", "path": ["h", "o", "c", "b", "a", "k", "f"], "base_speed": 17.9},
        {"name": "Route 2", "path": ["h", "p", "c", "b", "m", "f"], "base_speed": 26.5},
        {"name": "Route 3", "path": ["h", "j", "g", "e", "f"], "base_speed": 9.2}
    ]
}

@app.post("/analyze")
async def analyze_traffic(request: TrafficRequest):
    with open("traffic_data.json", "r") as f:
        traffic_db = json.load(f)

    stats = None
    for entry in traffic_db:
        if (entry["source"].upper() == request.source.upper() and 
            entry["destination"].upper() == request.destination.upper() and 
            entry["traffic_level"].upper() == request.traffic_level.upper()):
            stats = entry
            break

    if not stats:
        return {"explanation": "No matching data found in traffic_data.json."}

    # --- Step 2: Math Calculation ---
    vehicle_counts = {"LOW": 5, "MEDIUM": 15, "HIGH": 45}
    max_capacity = 50 
    current_vehicles = vehicle_counts.get(stats["traffic_level"].upper(), 0)
    congestion_index = current_vehicles / max_capacity 

    # --- Step 3: Decision Support Logic (LLM Integration) ---
    route_key = f"{request.source.lower()}-{request.destination.lower()}"
    available_routes = ROUTE_OPTIONS.get(route_key, [{"name": "Default", "path": [request.source, request.destination], "base_speed": 10}])

    # Build the Prompt for Llama
    prompt = f"""
    You are a SUMO Traffic AI Assistant.
    
    RULES:
    1. If Vehicle='Ambulance', prioritize the route with the lowest Congestion Index (C).
    2. If Objective='Fastest route', choose the route with the highest Average Speed.
    
    CURRENT SITUATION:
    - Source: {request.source}
    - Destination: {request.destination}
    - Vehicle Type: {request.vehicle_type}
    - User Objective: {request.objective}
    - Congestion Index (C): {congestion_index:.2f}
    - Traffic Level: {stats['traffic_level']}
    
    AVAILABLE ROUTES (JSON):
    {json.dumps(available_routes)}

    DECISION TASK:
    Analyze the situation. Tell me which Route Name to pick and provide a short one-sentence explanation.
    
    OUTPUT FORMAT:
    Route: [Route Name]
    Reasoning: [Explanation]
    """

    try:
        # Calling Llama via LM Studio
        response = client.chat.completions.create(
            model="lmstudio-community/meta-llama-3-8b-instruct", 
            messages=[{"role": "system", "content": "You are a professional traffic analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        llm_output = response.choices[0].message.content
        
        # Simple parsing logic for the LLM response
        # In a production app, you might want to ask the LLM for JSON output
        chosen_path = available_routes[0]["path"] # Default
        for route in available_routes:
            if route["name"].lower() in llm_output.lower():
                chosen_path = route["path"]
                break

    except Exception as e:
        llm_output = f"Error connecting to LM Studio: {str(e)}"
        chosen_path = available_routes[0]["path"]

    return {
        "recommended_route": " -> ".join(chosen_path),
        "travel_time": f"{stats['total_travel_time_sec']}s",
        "congestion": stats["traffic_level"],
        "speed": f"{stats['avg_speed_mps']} m/s",
        "explanation": llm_output
    }