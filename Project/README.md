<h1>Complete Command Reference — Smart Industrial Automation</h1>

---
### 1. sensor_simulator.py — Sensor Data Generator
Basic run (generate all 4 scenarios, save CSVs):
    python sensor_simulator.py

All argument variants:

# Generate all scenarios (default: 5s, 5000 Hz)
    python sensor_simulator.py --scenario all

---

# Generate a single scenario
python sensor_simulator.py --scenario normal
python sensor_simulator.py --scenario bearing_fault
python sensor_simulator.py --scenario stator_fault
python sensor_simulator.py --scenario rotor_bar_fault

# Change duration (seconds)
python sensor_simulator.py --scenario all --duration 10
python sensor_simulator.py --scenario bearing_fault --duration 30

# Change sampling frequency
python sensor_simulator.py --scenario all --fs 1000
python sensor_simulator.py --scenario all --fs 10000

# Change output directory for CSVs
python sensor_simulator.py --scenario all --output-dir ./data

# Print extracted feature JSON to console
python sensor_simulator.py --scenario bearing_fault --features

# Combined: all scenarios, long duration, high fs, features printed
python sensor_simulator.py --scenario all --duration 10 --fs 5000 --output-dir ./output --features

---

### 2. can_node.py — CAN FD Transmitter Node
Basic run (healthy/normal, 1 frame/sec):
bashpython can_node.py
All argument variants:
bash# Fault scenarios (friendly alias names)
python can_node.py --fault none            # normal operation
python can_node.py --fault bearing_wear    # → bearing_fault
python can_node.py --fault overheating     # → stator_fault
python can_node.py --fault motor_overload  # → rotor_bar_fault
---
# Change transmit interval (seconds between frames)
python can_node.py --fault none --interval 0.5
python can_node.py --fault bearing_wear --interval 2.0

# Change sensor data generation duration
python can_node.py --fault overheating --duration 30

# Change sampling frequency of simulated data
python can_node.py --fault none --fs 5000

# Combined: bearing fault, fast frames, long run
python can_node.py --fault bearing_wear --interval 0.5 --duration 60 --fs 1000

Note: can_node.py requires ml_gateway.py to be running first (Flask API on port 5000), as it transmits frames to the gateway.

---

### 3. ml_gateway.py — ML Gateway + Flask REST API
Run the gateway (starts Flask on port 5000):
bashpython ml_gateway.py
No command-line arguments — it always binds to 0.0.0.0:5000. After it starts, these API endpoints are live in your browser or with curl:
bash# Live system status (both motors)
curl http://localhost:5000/api/status

---

# Individual motor state
curl http://localhost:5000/api/motor/1
curl http://localhost:5000/api/motor/2

# Historical readings (rolling buffer)
curl http://localhost:5000/api/history

# Anomaly alert log
curl http://localhost:5000/api/alerts

---

### 4. run_demo.py — Single-Terminal Integration Demo
Full default demo (all 4 scenarios, 30s each, ~2 min total):
bashpython run_demo.py
All argument variants:
bash# Change dwell time per scenario (seconds)
python run_demo.py --dwell 60          # slow examiner demo (~4 min)
python run_demo.py --dwell 5           # fast CI smoke-test (~20s total)
python run_demo.py --dwell 10          # quick showcase (~40s)

---

# Change frame interval (seconds between data frames)
python run_demo.py --interval 0.1      # fast stream (10 frames/sec)
python run_demo.py --interval 1.0      # slow stream (1 frame/sec)
python run_demo.py --interval 2.0      # very slow (examiner can read each line)

# Run only specific scenarios
python run_demo.py --scenarios "normal,bearing_fault"
python run_demo.py --scenarios "stator_fault,rotor_bar_fault"
python run_demo.py --scenarios "bearing_fault"

# Skip auto-opening dashboard.html in browser
python run_demo.py --no-browser

# Recommended combinations:
# Viva / examiner presentation — slow, all faults visible
python run_demo.py --dwell 60 --interval 1.0

# CI pipeline / quick test — fast, minimal output
python run_demo.py --dwell 5 --interval 0.1 --no-browser

# Show only fault scenarios, skip normal
python run_demo.py --scenarios "bearing_fault,stator_fault,rotor_bar_fault" --dwell 30

# Two-fault demo with slow frames, no browser
python run_demo.py --scenarios "normal,stator_fault" --dwell 45 --interval 0.5 --no-browser

---

### 5. demo_report.py — HTML Evidence Report Generator

Run this while run_demo.py is active (Flask must be live on port 5000).

Default run (connects to localhost:5000, saves demo_report.html):
bashpython demo_report.py
All argument variants:
bash# Custom API host (e.g., Raspberry Pi on your LAN)
python demo_report.py --host 192.168.1.50

---

# Custom port
python demo_report.py --port 8080

# Custom output file path
python demo_report.py --output my_evidence_report.html
python demo_report.py --output ./reports/viva_report.html

# Skip auto-opening the report in browser
python demo_report.py --no-browser

# Remote Pi, custom port, custom output, no browser
python demo_report.py --host 192.168.1.50 --port 5000 --output pi_report.html --no-browser

---

### 6. Test Suite — pytest
bash# Run all 181 tests from project root
python -m pytest tests/

---

# Run with verbose output (shows each test name)
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_sensor.py
python -m pytest tests/test_canfd.py
python -m pytest tests/test_ml.py
python -m pytest tests/test_api.py

# Run tests matching a keyword
python -m pytest tests/ -k "bearing"
python -m pytest tests/ -k "anomaly"

# Show print output during tests
python -m pytest tests/ -s

# Stop after first failure
python -m pytest tests/ -x

# Short summary of failures only
python -m pytest tests/ -q

---

### Recommended Demo Workflow for Viva

---

bash# Terminal 1 — start the full demo
python run_demo.py --dwell 60 --interval 1.0

---

# Terminal 2 — (while demo runs) generate the evidence report
python demo_report.py --output viva_evidence.html
---