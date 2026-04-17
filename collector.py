import traci
import json
import time

# List your specific source and destination nodes here
SOURCES = ["K", "L", "M", "N", "D", "E"]
DESTINATIONS = ["H", "J"]

def start_collection():
    # 1. Start SUMO (Replace 'map.sumocfg' with your filename)
    # Use 'sumo-gui' to see the cars, or 'sumo' for faster collection
    traci.start(["sumo-gui", "-c", "map.sumocfg"]) 

    print("Simulation started. Collecting data...")

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep() # Move the simulation 1 step forward
            
            traffic_snapshot = []
            all_edges = traci.edge.getIDList()

            for edge in all_edges:
                if ":" in edge: continue # Skip junction internals

                # 2. Capture the data for this road segment
                speed = traci.edge.getLastStepMeanSpeed(edge) * 3.6 # m/s to km/h
                travel_time = traci.edge.getTraveltime(edge)
                occupancy = traci.edge.getLastStepOccupancy(edge) # 0 to 1

                # 3. Simple Safety/Congestion Logic
                status = "Clear"
                if occupancy > 0.7: status = "Heavy Traffic (Unsafe)"
                elif occupancy > 0.3: status = "Moderate Traffic"

                traffic_snapshot.append({
                    "road_id": edge,
                    "speed_kmh": round(speed, 2),
                    "time_sec": round(travel_time, 1),
                    "condition": status
                })

            # 4. Save to JSON so FastAPI can read it
            with open("simulation_data.json", "w") as f:
                json.dump(traffic_snapshot, f, indent=4)
            
            time.sleep(0.1) # Small pause to avoid crashing your CPU
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        traci.close()

if __name__ == "__main__":
    start_collection()