import json

def get_llm_input(level_name, vehicle="Car", objective="Fastest route"):
    # Load your existing JSON data
    try:
        with open("traffic_data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return "Error: traffic_data.json not found."

    # Find the specific traffic level data
    # Matches "HIGH", "MEDIUM", or "LOW" from your screenshot
    record = next((item for item in data if item["traffic_level"] == level_name.upper()), None)

    if not record:
        return f"Error: No data found for {level_name}"

    # Construct the Pure Input Object
    pure_inputs = {
        "Traffic": record["traffic_level"].capitalize(),
        "Speed": record["avg_speed_mps"],
        "Vehicle_Type": vehicle,
        "Objective": objective
    }
    
    return pure_inputs

# Example: Prepare input for an Ambulance in High Traffic
final_input = get_llm_input("HIGH", vehicle="Ambulance", objective="Fastest route")
print(json.dumps(final_input, indent=4))