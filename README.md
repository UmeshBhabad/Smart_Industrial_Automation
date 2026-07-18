<h1 align="center">⚙️ Smart Industrial Automation</h1>

<h3 align="center">Motor Fault Detection via CAN FD Bus · Isolation Forest ML · Flask REST API · Live Dashboard</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/Flask-REST%20API-lightgrey?logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/ML-Isolation%20Forest-orange?logo=scikit-learn" alt="Isolation Forest"/>
  <img src="https://img.shields.io/badge/Bus-CAN%20FD-green" alt="CAN FD"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License"/>
</p>

<hr>

## 📌 Project Overview

This project implements a **Predictive Maintenance (PdM) system** for industrial motors. Sensor data from two motors — a **Three-Phase Induction Motor** and a **Centrifugal Pump Motor** — is streamed over a simulated **CAN FD bus**, analysed in real-time by an **Isolation Forest** anomaly detector running on a Raspberry Pi gateway, and displayed on a **live browser dashboard** with email and browser alert capabilities.

The system detects four operating conditions:

| Condition | Description |
|---|---|
| `normal` | Healthy baseline — all sensors in nominal range |
| `bearing_fault` | Outer-race impact impulses → elevated vibration and temperature |
| `stator_fault` | Turn-to-turn short → I²R hotspot, unbalanced phase currents |
| `rotor_bar_fault` | Broken rotor bars → torque pulsation, speed drop, current sidebands |

<hr>

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SENSOR SIMULATION LAYER                             │
│                                                                              │
│   sensor_simulator.py                                                        │
│   ┌────────────────────────┐   ┌────────────────────────┐                    │
│   │  Motor 1               │   │  Motor 2               │                    │
│   │  Three-Phase Induction │   │  Centrifugal Pump      │                    │
│   │  temp · vib · speed    │   │  temp · vib · flow     │                    │
│   │  Ia · Ib · Ic · sound  │   │  Ia · Ib · Ic · sound  │                    │
│   └───────────┬────────────┘   └────────────┬───────────┘                    │
└───────────────┼─────────────────────────────┼────────────────────────────────┘
                │                             │
                ▼                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CAN FD BUS LAYER                                   │
│                                                                              │
│   can_node.py  (simulates ESP32 WROOM-32 transmitter nodes)                  │
│                                                                              │
│   Motor 1 PRIMARY   frame  ID=0x100  32 bytes  '<ffffffBxxxxxxx'             │
│   Motor 1 SECONDARY frame  ID=0x101   8 bytes  '<ff'                         │
│   Motor 2 PRIMARY   frame  ID=0x200  32 bytes  '<ffffffBxxxxxxx'             │
│   Motor 2 SECONDARY frame  ID=0x201   8 bytes  '<ff'                         │
│                                                                              │
│   Transport: python-can  udp_multicast virtual bus                           │
│   Bit-rate: 500 kbit/s arbitration · 2 Mbit/s data phase (CAN FD)            │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ML GATEWAY LAYER  (Raspberry Pi 4)                    │
│                                                                              │
│   ml_gateway.py                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  Thread 1 (daemon): CAN FD Listener                                  │   │
│   │    unpack struct bytes → score with Isolation Forest → update state  │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  Thread 2 (main): Flask REST API                                     │   │
│   │    GET  /api/status      GET  /api/motor/1    GET  /api/motor/2      │   │
│   │    GET  /api/history     GET  /api/alerts     POST /api/email_alert  │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│   Anomaly thresholds:                                                        │
│     score > 0.0    → NORMAL    score ∈ (-0.10, 0.0] → WARNING                │
│     score < -0.10  → CRITICAL                                                │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │  HTTP polling (1–2 s interval)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        DASHBOARD LAYER  (Any browser on LAN)                 │
│                                                                              │
│   dashboard.html  — single-file, no build step                               │
│   ┌───────────┐  ┌──────────────────┐  ┌────────────┐  ┌─────────────────┐   │
│   │ Overview  │  │  Sensor Trends   │  │  Analysis  │  │     Alerts      │   │
│   │ anomaly   │  │  temp · vib      │  │  radar     │  │  log table      │   │
│   │ gauge     │  │  speed/flow      │  │  scatter   │  │  CSV download   │   │
│   │ health    │  │  humidity        │  │  bar       │  │  email config   │   │
│   │ ring PdM  │  │  anomaly score   │  │  donut     │  │  browser notif  │   │
│   │ estimate  │  │  3-phase current │  │  charts    │  │  audio beep     │   │
│   └───────────┘  └──────────────────┘  └────────────┘  └─────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

<hr>

## 📂 File Structure

```
Smart_Industrial_Automation/
│
├── Documentations/               # Project documentation files
├── Project/                      # Core project source files
|    ├── sensor_simulator.py   # Motor sensor data generator (4 fault scenarios × 2 motors)
|    ├── can_node.py           # ESP32 CAN FD transmitter node simulator
|    ├── ml_gateway.py         # Raspberry Pi gateway: CAN listener + ML + Flask REST API
|    ├── dashboard.html        # Live browser monitoring dashboard (Chart.js, no build step)
|    ├── run_demo.py           # Integration demo orchestrator (single-terminal, no hardware)
|    ├── demo_report.py        # Evidence report generator (HTML with Chart.js charts)
|    ├── requirements.txt      # Pinned Python dependencies
│
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

<hr>

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Sensor simulation | Python · NumPy · pandas · SciPy |
| CAN FD bus | python-can (udp_multicast virtual bus) |
| ML anomaly detection | scikit-learn Isolation Forest |
| REST API | Flask · flask-cors |
| Email alerts | smtplib (Python stdlib — no extra install) |
| Dashboard | HTML5 · Chart.js · Web Notifications API · Web Audio API |
| Target hardware | ESP32 WROOM-32 (CAN node) · Raspberry Pi 4 (gateway) |

<hr>

## ✅ Prerequisites

- Python **3.8** or newer
- `pip` package manager
- A modern browser (Chrome, Edge, or Firefox) for the dashboard
- *(Optional)* Gmail account with an [App Password](https://myaccount.google.com/apppasswords) for email alerts

> **No physical hardware required.** The demo runs entirely on software-simulated CAN FD bus using the `udp_multicast` virtual interface built into python-can.

<hr>

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-industrial-automation.git
cd smart-industrial-automation
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

<hr>

## ▶️ Usage

There are two ways to run the system: the **quick demo** (recommended for evaluation) or the **full hardware-style pipeline**.

---

### Option A — Quick Demo (Recommended)

Runs all four fault scenarios automatically in a single terminal. No virtual CAN bus setup needed.

```bash
python run_demo.py
```

**What happens:**
1. Both Isolation Forest models are trained on simulated normal data
2. Flask REST API starts on `http://localhost:5000`
3. All four fault scenarios cycle automatically, printing live console output
4. A summary table is displayed at the end
5. The API stays alive — open `dashboard.html` in your browser to inspect live data

**Optional flags:**

```bash
python run_demo.py --frames 30          # frames per scenario (default: 20)
python run_demo.py --interval 0.5       # seconds between frames (default: 0.3)
python run_demo.py --open-browser       # auto-open dashboard.html
python run_demo.py --motor 1            # run only Motor 1 scenarios
```

---

### Option B — Evidence Report

While `run_demo.py` is running, open a second terminal and run:

```bash
python demo_report.py
```

This queries all five API endpoints and generates `demo_report.html` — a self-contained, offline-viewable HTML report with Chart.js charts, alert logs, and raw API JSON responses. It is automatically opened in your browser.

```bash
# Point to a Raspberry Pi on your LAN instead of localhost
python demo_report.py --host 192.168.1.50 --port 5000
```

---

### Option C — Full CAN FD Pipeline (Hardware / Virtual Bus)

Run each component in a separate terminal:

**Terminal 1 — ML Gateway:**
```bash
python ml_gateway.py
```

**Terminal 2 — CAN FD Transmitter Node:**
```bash
# Transmit all scenarios on repeat (10 s each)
python can_node.py --scenario all

# Transmit a specific scenario
python can_node.py --scenario bearing_fault --duration 30

# Simulate Motor 2 (pump)
python can_node.py --motor 2 --scenario stator_fault
```

**Terminal 3 — Dashboard:**
Open `dashboard.html` directly in your browser. No web server is needed — it polls `http://localhost:5000` automatically.

---

### Sensor Data Only

Generate raw CSV sensor data without running the full pipeline:

```bash
# Generate all 4 scenarios to CSV files
python sensor_simulator.py --scenario all --duration 10 --output-dir ./data

# Generate one scenario and print extracted features
python sensor_simulator.py --scenario bearing_fault --features
```

<hr>

## 🌐 API Reference

All endpoints are served by `ml_gateway.py` on `http://localhost:5000` by default.

---

### `GET /api/status`

Returns the latest sensor readings and anomaly status for **both** motors. Used by the dashboard Overview tab.

**Example response:**
```json
{
  "gateway_start": "2024-01-15T10:22:03.000",
  "timestamp": "2024-01-15T10:22:45.312",
  "motor1": {
    "temperature_C": 87.4,
    "vibration_x_g": 0.412,
    "speed_rpm": 1470.2,
    "current_a_A": 41.3,
    "current_b_A": 39.1,
    "current_c_A": 40.7,
    "ambient_temp_C": 26.1,
    "humidity_pct": 52.3,
    "fault_id": 1,
    "fault_name": "BEARING_FAULT",
    "anomaly_score": -0.142,
    "anomaly_status": "CRITICAL"
  },
  "motor2": { "..." : "..." }
}
```

---

### `GET /api/motor/1`

Full detail snapshot for **Motor 1** (Three-Phase Induction Motor).

---

### `GET /api/motor/2`

Full detail snapshot for **Motor 2** (Centrifugal Pump Motor). Uses `flow_rate_Lm` instead of `speed_rpm`.

---

### `GET /api/history`

Returns the last **60 readings** per motor for Chart.js trend charts.

**Response fields per entry:** `time`, `temperature_C`, `vibration_x_g`, `current_a_A`, `current_b_A`, `current_c_A`, `anomaly_score`, `anomaly_status`, `speed_rpm` (Motor 1) or `flow_rate_Lm` (Motor 2).

---

### `GET /api/alerts`

List of recent `WARNING` and `CRITICAL` anomaly events, newest first.

**Example response:**
```json
{
  "timestamp": "2024-01-15T10:22:45.312",
  "count": 3,
  "alerts": [
    {
      "motor": "motor1",
      "status": "CRITICAL",
      "fault_name": "BEARING_FAULT",
      "anomaly_score": -0.142,
      "temperature_C": 87.4,
      "vibration_x_g": 0.412,
      "current_a_A": 41.3,
      "timestamp": "10:22:44.891"
    }
  ]
}
```

---

### `POST /api/email_alert`

Sends a fault alert email via SMTP. Called automatically by the dashboard when a `WARNING` or `CRITICAL` event fires, or manually from the Alerts tab.

**Request body:**
```json
{
  "recipient":     "engineer@plant.com",
  "motor":         "motor1",
  "status":        "CRITICAL",
  "fault_name":    "BEARING_FAULT",
  "anomaly_score": -0.142,
  "temperature_C": 87.4,
  "vibration_x_g": 0.412,
  "current_a_A":   41.3,
  "timestamp":     "10:22:44.891"
}
```

**Returns:** `200 OK` on success · `503` if SMTP not configured · `400` on bad request.

---

### Anomaly Score Interpretation

| `anomaly_score` range | `anomaly_status` | Meaning |
|---|---|---|
| `> 0.0` | `NORMAL` | Far from anomaly boundary |
| `-0.10` to `0.0` | `WARNING` | Borderline — monitor closely |
| `< -0.10` | `CRITICAL` | Clearly anomalous — act immediately |

<hr>

## 📧 Email Alert Configuration

Set these environment variables **before** launching `ml_gateway.py` or `run_demo.py`:

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=your@gmail.com
export SMTP_PASS=your_app_password     # Gmail App Password, not account password
export SMTP_FROM=your@gmail.com
```

> **Gmail users:** Enable 2-Factor Authentication, then create an App Password at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords). Do not use your regular Gmail password.

<hr>

## 🔩 CAN FD Frame Format

Both `can_node.py` and `ml_gateway.py` use identical struct definitions. The little-endian byte order (`<`) matches the ESP32's native endianness (Xtensa LX6), allowing a real embedded node to `memcpy()` bytes directly into a C `float[]`.

```
PRIMARY frame (32 bytes) — IDs: 0x100 (Motor 1), 0x200 (Motor 2)
Format: '<ffffffBxxxxxxx'

  Bytes  0– 3 : temperature_C       (float32, °C)
  Bytes  4– 7 : vibration_x_g       (float32, g)
  Bytes  8–11 : speed_rpm / flow_Lm (float32)
  Bytes 12–15 : current_a_A         (float32, A)
  Bytes 16–19 : current_b_A         (float32, A)
  Bytes 20–23 : current_c_A         (float32, A)
  Byte  24    : fault_id            (uint8, 0–3)
  Bytes 25–31 : padding             (0x00)

SECONDARY frame (8 bytes) — IDs: 0x101 (Motor 1), 0x201 (Motor 2)
Format: '<ff'

  Bytes 0–3 : ambient_temp  (float32, °C)
  Bytes 4–7 : humidity      (float32, %)
```

<hr>

## 🧠 ML Model Details

The anomaly detector uses **scikit-learn's Isolation Forest**, one per motor.

- **Training data:** 200 synthetic samples generated from the `normal` scenario of `sensor_simulator.py`
- **Feature vector (3 features per motor):** `temperature_C`, `vibration_x_g`, `speed_rpm` / `flow_rate_Lm`
- **Contamination parameter:** 5% (`contamination=0.05`)
- **Inference:** `decision_function()` score is computed per frame and returned alongside every API response
- **No persistence:** models retrain from scratch each time `ml_gateway.py` or `run_demo.py` starts — training takes < 1 second

<hr>

## 📊 Dashboard Tabs

| Tab | Contents |
|---|---|
| **Overview** | Anomaly score gauge, health ring chart, PdM remaining-life estimate, last-known sensor readings for both motors |
| **Sensor Trends** | Live Chart.js line charts: temperature, vibration, speed/flow, humidity, anomaly score, 3-phase current overlay (Ia, Ib, Ic) |
| **Analysis** | Radar chart (sensor profile), scatter (vibration vs temperature), bar (fault distribution), donut (status breakdown) |
| **Alerts** | Alert log table with timestamps, CSV download button, email recipient configuration, test-alert button |

The dashboard also produces **browser push notifications** and a **Web Audio API beep** when a `CRITICAL` anomaly is detected (requires notification permission).

<hr>

## 🔮 Future Enhancements

- Persistent model storage (joblib / pickle) — avoid retraining on every restart
- MQTT broker integration for broader IoT compatibility
- Multi-node CAN FD topology (more than two motors)
- Real-time FFT spectrum view in the dashboard
- Docker Compose setup for one-command deployment
- REST API authentication (API key or JWT)
- Unit test suite (pytest) with CI via GitHub Actions


## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<hr>

## ⭐ Support

If you find this project useful, please consider giving it a ⭐ on GitHub!
