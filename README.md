# Oval Telemetry Simulator - Real-Time Vehicle Telemetry Generator

A python simulation of a race car lapping a simple "oval track" with straights and corners.
The simulator generates live telemetry data (speed, RPM, throttle, gear, lap times, etc.),
visualizes it in real time with Matplotlib, and logs the session to CSV for later analysis.

This project can be used for:
- Practicing with telemetry visualization & dashboards  
- Generating sample datasets for motorsport/data engineering pipelines  
- Demonstrating simulation, control, and dynamics modeling

Short paragraph: what this project does, who it’s for, and what makes it interesting.

## Demo
When running the simulator, you’ll see:  
- **Live speed plot** vs. time  
- **Live RPM plot** vs. time  
- **Dashboard panel** showing lap time, estimated lap time, current gear, throttle, and session metrics  

CSV logs look like this (first few lines):

```csv
time_s,lap,lap_time_s,distance_m,distance_into_lap_m,segment,segment_type,speed_kph,rpm,throttle,gear,avg_speed_session_kph,max_rpm_session,est_lap_time_s
0.050,1,0.050,0.17,0.17,Straight A,straight,12.15,1683,0.330,1,12.15,1683,
0.100,1,0.100,0.53,0.53,Straight A,straight,18.95,1983,0.420,1,15.55,1983,
```

## Quick Start
1. Clone this repo
git clone https://github.com/taepop/racecar_telemetry_sim.git
cd racecar_telemetry_sim

2. Install requirements
pip install matplotlib numpy

3. Run the simulator
python oval_telemetry.py --hz 20 --csv telemetry.csv

oval-telemetry-sim/
│
├── telemetry_sim.py     # Main simulation + visualization
├── telemetry.csv        # Sample CSV output (generated at runtime)
└── README.md            # Project documentation


## Results / What I learned
- Modeled a simplified car dynamics pipeline (throttle -> force -> acceleration -> speed)
- Built a real-time telemetry visualization dashboard with Python + Matplotlib
- Practiced simulation loops with control logic (throttle controller + gearbox)
- Learned about data logging pipelines in motorsport context
 
