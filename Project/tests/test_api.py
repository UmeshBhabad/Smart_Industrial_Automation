"""
test_api.py — Flask REST API Validation (v2)
=============================================
Updated for the upgraded gateway which now stores and exposes
three-phase stator currents (current_a_A, current_b_A, current_c_A)
in all motor endpoints and history entries.

New fields verified in v2:
  /api/status    → motor1/motor2 now include current_a_A, current_b_A, current_c_A
  /api/motor/1   → full detail includes all three current fields
  /api/motor/2   → same
  /api/history   → each history entry now includes current_a_A, current_b_A, current_c_A
  /api/alerts    → each alert entry now includes current_a_A

WHY Flask test client over a live server?
  In-process WSGI dispatch: no TCP, no port conflicts, ~100× faster,
  deterministic. Equivalent to FAT Level 3 — API Contract Test.
"""

import pytest
import json

VALID_STATUSES = {"NORMAL", "WARNING", "CRITICAL", "INITIALISING"}

# ── ALL required keys for /api/motor/1 and /api/motor/2 ─────────────────────
MOTOR1_REQUIRED_KEYS = {
    "motor_id", "type", "timestamp",
    "temperature_C", "vibration_x_g",
    "current_a_A", "current_b_A", "current_c_A",   # NEW in v2
    "speed_rpm", "flow_rate_Lm",
    "ambient_temp_C", "humidity_pct",
    "fault_id", "fault_name",
    "anomaly_score", "anomaly_status",
    "last_updated", "rx_count",
}

# Keys expected inside each /api/history entry
HISTORY_ENTRY_KEYS = {
    "time", "temperature_C", "vibration_x_g",
    "current_a_A", "current_b_A", "current_c_A",   # NEW in v2
    "anomaly_score", "anomaly_status",
}

# Keys expected inside each alert entry
ALERT_ENTRY_KEYS = {
    "timestamp", "motor", "status", "anomaly_score",
    "fault_name", "temperature_C", "vibration_x_g",
    "current_a_A",   # NEW in v2
}


def _json(response):
    return json.loads(response.data.decode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# 1. GET /api/status
# ─────────────────────────────────────────────────────────────────────────────

class TestApiStatus:

    def test_200(self, flask_test_client):
        assert flask_test_client.get("/api/status").status_code == 200

    def test_top_level_keys(self, flask_test_client):
        data    = _json(flask_test_client.get("/api/status"))
        missing = {"motor1", "motor2", "gateway_start", "timestamp"} - set(data)
        assert not missing, f"Missing keys: {missing}"

    def test_motor_key_spelling(self, flask_test_client):
        """motor1/motor2, NOT motor_1/motor_2."""
        data = _json(flask_test_client.get("/api/status"))
        assert "motor_1" not in data and "motor_2" not in data
        assert "motor1" in data and "motor2" in data

    def test_motor1_anomaly_status_valid(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        assert data["motor1"]["anomaly_status"] in VALID_STATUSES

    def test_motor2_anomaly_status_valid(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        assert data["motor2"]["anomaly_status"] in VALID_STATUSES

    def test_motor1_has_current_fields(self, flask_test_client):
        """UPGRADED: /api/status motor1 must include three-phase currents."""
        data = _json(flask_test_client.get("/api/status"))
        m1   = data["motor1"]
        for field in ("current_a_A", "current_b_A", "current_c_A"):
            assert field in m1, f"motor1 missing '{field}' in /api/status"

    def test_motor2_has_current_fields(self, flask_test_client):
        """UPGRADED: /api/status motor2 must include three-phase currents."""
        data = _json(flask_test_client.get("/api/status"))
        m2   = data["motor2"]
        for field in ("current_a_A", "current_b_A", "current_c_A"):
            assert field in m2, f"motor2 missing '{field}' in /api/status"

    def test_current_fields_are_none_or_numeric(self, flask_test_client):
        """
        On fresh gateway (no CAN frames received) currents are None.
        After frames arrive they become floats. Both are valid.
        """
        data = _json(flask_test_client.get("/api/status"))
        for motor in ("motor1", "motor2"):
            for field in ("current_a_A", "current_b_A", "current_c_A"):
                val = data[motor][field]
                assert val is None or isinstance(val, (int, float)), (
                    f"{motor}.{field} = {val!r} (expected None or numeric)")

    def test_rx_count_is_integer(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        assert isinstance(data["motor1"]["rx_count"], int)
        assert isinstance(data["motor2"]["rx_count"], int)

    def test_gateway_start_is_string(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        assert isinstance(data["gateway_start"], str)

    def test_timestamp_is_string(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        assert isinstance(data["timestamp"], str)

    def test_consistent_across_calls(self, flask_test_client):
        d1 = _json(flask_test_client.get("/api/status"))
        d2 = _json(flask_test_client.get("/api/status"))
        assert set(d1.keys()) == set(d2.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 2. GET /api/motor/1
# ─────────────────────────────────────────────────────────────────────────────

class TestApiMotor1:

    def test_200(self, flask_test_client):
        assert flask_test_client.get("/api/motor/1").status_code == 200

    def test_motor_id_is_1(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert data["motor_id"] == 1

    def test_type_is_induction_motor(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert data["type"] == "Three-Phase Induction Motor"

    def test_all_required_keys(self, flask_test_client):
        data    = _json(flask_test_client.get("/api/motor/1"))
        missing = MOTOR1_REQUIRED_KEYS - set(data)
        assert not missing, (
            f"GET /api/motor/1 missing keys: {missing}\n"
            f"Got: {sorted(data.keys())}")

    def test_current_fields_present(self, flask_test_client):
        """UPGRADED: three-phase currents must be in the motor detail response."""
        data = _json(flask_test_client.get("/api/motor/1"))
        for field in ("current_a_A", "current_b_A", "current_c_A"):
            assert field in data, f"/api/motor/1 missing '{field}'"

    def test_anomaly_status_valid(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert data["anomaly_status"] in VALID_STATUSES

    def test_rx_count_is_integer(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert isinstance(data["rx_count"], int) and data["rx_count"] >= 0

    def test_temperature_is_none_or_numeric(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        t = data["temperature_C"]
        assert t is None or isinstance(t, (int, float))

    def test_timestamp_is_string(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert isinstance(data["timestamp"], str)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GET /api/motor/2
# ─────────────────────────────────────────────────────────────────────────────

class TestApiMotor2:

    def test_200(self, flask_test_client):
        assert flask_test_client.get("/api/motor/2").status_code == 200

    def test_motor_id_is_2(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/2"))
        assert data["motor_id"] == 2

    def test_type_is_pump_motor(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/2"))
        assert data["type"] == "Centrifugal Pump Motor"

    def test_has_flow_rate_key(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/2"))
        assert "flow_rate_Lm" in data

    def test_anomaly_status_valid(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/2"))
        assert data["anomaly_status"] in VALID_STATUSES

    def test_current_fields_present(self, flask_test_client):
        """UPGRADED: pump motor also transmits three-phase currents."""
        data = _json(flask_test_client.get("/api/motor/2"))
        for field in ("current_a_A", "current_b_A", "current_c_A"):
            assert field in data, f"/api/motor/2 missing '{field}'"


# ─────────────────────────────────────────────────────────────────────────────
# 4. GET /api/history
# ─────────────────────────────────────────────────────────────────────────────

class TestApiHistory:

    def test_200(self, flask_test_client):
        assert flask_test_client.get("/api/history").status_code == 200

    def test_top_level_keys(self, flask_test_client):
        data    = _json(flask_test_client.get("/api/history"))
        missing = {"motor1", "motor2", "timestamp"} - set(data)
        assert not missing

    def test_motor1_is_list(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        assert isinstance(data["motor1"], list)

    def test_motor2_is_list(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        assert isinstance(data["motor2"], list)

    def test_history_length_le_60(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        assert len(data["motor1"]) <= 60
        assert len(data["motor2"]) <= 60

    def test_history_entry_keys_if_populated(self, flask_test_client):
        """UPGRADED: history entries now include current_a/b/c_A."""
        data = _json(flask_test_client.get("/api/history"))
        for entry in data["motor1"]:
            missing = HISTORY_ENTRY_KEYS - set(entry)
            assert not missing, (
                f"History entry missing keys: {missing}\nGot: {list(entry.keys())}")

    def test_history_anomaly_status_valid(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        for entry in data["motor1"]:
            assert entry["anomaly_status"] in VALID_STATUSES

    def test_history_current_fields_none_or_numeric(self, flask_test_client):
        """Current fields in history entries must be None or numeric."""
        data = _json(flask_test_client.get("/api/history"))
        for entry in data["motor1"]:
            for field in ("current_a_A", "current_b_A", "current_c_A"):
                if field in entry:
                    val = entry[field]
                    assert val is None or isinstance(val, (int, float)), (
                        f"History {field} = {val!r}")

    def test_timestamp_is_string(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        assert isinstance(data["timestamp"], str)


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET /api/alerts
# ─────────────────────────────────────────────────────────────────────────────

class TestApiAlerts:

    def test_200(self, flask_test_client):
        assert flask_test_client.get("/api/alerts").status_code == 200

    def test_response_is_dict_not_list(self, flask_test_client):
        """
        CRITICAL: must be a dict {timestamp, count, alerts:[...]}.
        A bare list would break the dashboard which reads response["count"].
        """
        data = _json(flask_test_client.get("/api/alerts"))
        assert isinstance(data, dict), (
            f"GET /api/alerts returned {type(data).__name__}, expected dict")

    def test_required_keys(self, flask_test_client):
        data    = _json(flask_test_client.get("/api/alerts"))
        missing = {"timestamp", "count", "alerts"} - set(data)
        assert not missing

    def test_alerts_is_list(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        assert isinstance(data["alerts"], list)

    def test_count_equals_alerts_length(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        assert data["count"] == len(data["alerts"])

    def test_count_is_non_negative_int(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        assert isinstance(data["count"], int) and data["count"] >= 0

    def test_alerts_empty_on_fresh_gateway(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        assert data["count"] == 0 and data["alerts"] == []

    def test_alert_entry_keys_if_populated(self, flask_test_client):
        """UPGRADED: alert entries now include current_a_A."""
        data = _json(flask_test_client.get("/api/alerts"))
        for alert in data["alerts"]:
            missing = ALERT_ENTRY_KEYS - set(alert)
            assert not missing, (
                f"Alert entry missing keys: {missing}\nGot: {list(alert.keys())}")

    def test_alert_status_valid_if_populated(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        for alert in data["alerts"]:
            assert alert["status"] in {"WARNING", "CRITICAL"}

    def test_timestamp_is_string(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        assert isinstance(data["timestamp"], str)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestApiEdgeCases:

    def test_invalid_motor_id_404(self, flask_test_client):
        assert flask_test_client.get("/api/motor/3").status_code == 404

    def test_unknown_route_404(self, flask_test_client):
        assert flask_test_client.get("/api/nonexistent").status_code == 404

    def test_root_404(self, flask_test_client):
        assert flask_test_client.get("/").status_code == 404

    def test_content_type_json(self, flask_test_client):
        for endpoint in ["/api/status", "/api/motor/1", "/api/motor/2",
                         "/api/history", "/api/alerts"]:
            r  = flask_test_client.get(endpoint)
            ct = r.content_type or ""
            assert "application/json" in ct, (
                f"GET {endpoint} Content-Type: '{ct}'")
