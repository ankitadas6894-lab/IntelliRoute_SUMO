import os
import sys
import traci

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def generate_routefile(num_vehicles):
    """Generates random traffic demand based on the count provided."""
    with open("road.rou.xml", "w") as routes:
        print("""<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>
    <route id="r1" edges="f_k k_a a_b b_c c_i i_j j_h"/>
    <route id="r2" edges="d_f f_m m_b b_n n_g g_e e_i"/>
    """, file=routes)
        
        for i in range(num_vehicles):
            # Alternate vehicles between two routes
            route = "r1" if i % 2 == 0 else "r2"
            print(f'    <vehicle id="veh{i}" type="car" route="{route}" depart="{i*2}"/>', file=routes)
        print("</routes>", file=routes)

def run_test(label, vehicle_count):
    generate_routefile(vehicle_count)
    
    # Start SUMO (use 'sumo-gui' instead of 'sumo' if you want to watch it)
    traci.start(["sumo-gui", "-c", "road.sumocfg"])
    
    step = 0
    total_speed = 0
    total_vehicles_tracked = 0
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        ids = traci.vehicle.getIDList()
        if ids:
            total_speed += sum([traci.vehicle.getSpeed(v) for v in ids])
            total_vehicles_tracked += len(ids)
        step += 1
    
    avg_speed = total_speed / total_vehicles_tracked if total_vehicles_tracked > 0 else 0
    # Congestion: 0 is free flow, 1 is gridlock
    congestion = max(0, 1 - (avg_speed / 13.89)) 

    print(f"\n--- {label} Traffic Simulation Results ---")
    print(f"Average Speed:    {avg_speed:.2f} m/s")
    print(f"Total Travel Time: {step} steps (seconds)")
    print(f"Congestion Level:  {congestion:.2f}")
    
    traci.close()

# Execute Scenarios
scenarios = {"LOW": 10, "MEDIUM": 50, "HIGH": 150}
for name, count in scenarios.items():
    run_test(name, count)