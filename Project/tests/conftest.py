"""
conftest.py — Shared pytest fixtures (v2 — 32-byte PRIMARY frame)
=================================================================
Upgraded to match the new CAN FD frame layout:
  PRIMARY_FMT = '<ffffffBxxxxxxx'  →  32 bytes
    6 × float32: temp, vib, speed/flow, current_a, current_b, current_c
    1 × uint8  : fault_id
    7 × padding

WHY FIXTURES instead of setUp/tearDown:
  • Composability  : each test file declares exactly the fixtures it needs.
  • Scope control  : scope="module" trains IsolationForest once per file,
                     not once per test — saves ~0.5 s × n_tests.
  • Yield cleanup  : setup + teardown in one function, no dual override.
  • Parameterisation: fixtures can be independently parameterised.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sensor_simulator import generate_scenario
from ml_gateway import MotorAnomalyDetector, app


# ── Sensor fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def motor1_simulator():
    """1 s of normal Motor 1 data at 1 kHz → 1000 rows."""
    return generate_scenario("normal", duration_s=1, fs=1000, seed=42)


@pytest.fixture(scope="module")
def motor2_simulator():
    """1 s of normal Motor 2 (pump) data — independent seed."""
    return generate_scenario("normal", duration_s=1, fs=1000, seed=99)


# ── ML model fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_detector1():
    """IsolationForest for Motor 1, trained on 200 normal samples."""
    d = MotorAnomalyDetector("motor1")
    d.train(n_samples=200)
    return d


@pytest.fixture(scope="module")
def trained_detector2():
    """Independent IsolationForest for Motor 2 (pump)."""
    d = MotorAnomalyDetector("motor2")
    d.train(n_samples=200)
    return d


# ── Flask test-client fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def flask_test_client():
    """
    In-process Flask client — no TCP socket, no network, deterministic.
    TESTING=True surfaces exceptions as Python tracebacks instead of
    HTML 500 pages, making test failures easier to diagnose.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
