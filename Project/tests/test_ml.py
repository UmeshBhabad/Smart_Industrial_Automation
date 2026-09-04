"""
test_ml.py — ML Anomaly Detection Validation (v2)
==================================================
Tests MotorAnomalyDetector (Isolation Forest wrapper).

IMPORTANT: The ML feature vector is UNCHANGED — still 3 channels:
  [temperature_C, vibration_x_g, speed_rpm/flow_rate]

The three-phase currents (Ia, Ib, Ic) are now transmitted, stored in
state, and displayed on the dashboard — but they are NOT yet used as
ML features. This is a deliberate design choice documented in Section VII
of the research paper (future work: add MCSA features).

What IS new in v2:
  • _handle_primary() takes Ia, Ib, Ic as extra arguments (gateway function)
  • _unpack_primary() returns 7 values (frame parser function)
  • The scoring function itself (detector.score()) is IDENTICAL to v1
  • Detection accuracy table now includes current RMS columns
    to report the sensor readings captured during each fault scenario

Research paper table outputs:
  TestDetectionAccuracyTable → Table III values
  TestFeatureAblation        → Table V values (per-channel detection)
  TestBaselineComparison     → Table VI values (model comparison)

WHY decision_function() not predict()?
  predict() returns binary +1/-1. decision_function() returns a continuous
  score giving WARNING/CRITICAL gradient thresholds and enabling ROC curves.
"""

import pytest
import numpy as np

from ml_gateway import (
    MotorAnomalyDetector,
    THRESHOLD_WARNING,
    THRESHOLD_CRITICAL,
    FAULT_NAMES,
    _unpack_primary,
)
from sensor_simulator import generate_scenario


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES = 100

def _score_scenario(detector, scenario, n=N_SAMPLES, seed=42):
    """
    Generate n samples of fault scenario, score each row,
    return (scores list, statuses list, mean_ia, mean_ib, mean_ic).
    Uses distinct random seed per call so samples are not temporally correlated.
    """
    duration = n / 1000.0 + 0.1
    df = generate_scenario(scenario, duration_s=duration, fs=1000, seed=seed)
    df = df.head(n)

    scores, statuses = [], []
    for _, row in df.iterrows():
        s, st = detector.score(
            row["temperature_C"],
            row["vibration_x_g"],
            row["speed_rpm"],
        )
        scores.append(s)
        statuses.append(st)

    return (scores, statuses,
            df["current_a_A"].values,
            df["current_b_A"].values,
            df["current_c_A"].values)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestModelLifecycle:

    def test_untrained_returns_initialising(self):
        d = MotorAnomalyDetector("motor1")
        score, status = d.score(65.0, 0.05, 1480.0)
        assert score  == 0.0,           f"Untrained score {score}, expected 0.0"
        assert status == "INITIALISING", f"Untrained status '{status}'"

    def test_trained_flag_before_train(self):
        assert MotorAnomalyDetector("motor1").trained is False

    def test_trained_flag_after_train(self):
        d = MotorAnomalyDetector("motor1")
        d.train(n_samples=50)
        assert d.trained is True

    def test_train_200_samples(self):
        d = MotorAnomalyDetector("motor1")
        d.train(n_samples=200)
        assert d.trained

    def test_motor2_train(self):
        d = MotorAnomalyDetector("motor2")
        d.train(n_samples=200)
        assert d.trained

    def test_motor_id_stored(self):
        d = MotorAnomalyDetector("motor1")
        assert d.motor_id == "motor1"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model independence
# ─────────────────────────────────────────────────────────────────────────────

class TestModelIndependence:

    def test_separate_model_objects(self, trained_detector1, trained_detector2):
        """
        Motor 1 (RPM features) and Motor 2 (flow_rate features) must have
        independent IsolationForest instances — sharing would contaminate
        the normal cluster boundary for both machines.
        """
        assert trained_detector1.model is not trained_detector2.model

    def test_different_motor_ids(self, trained_detector1, trained_detector2):
        assert trained_detector1.motor_id != trained_detector2.motor_id


# ─────────────────────────────────────────────────────────────────────────────
# 3. Normal data scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalScoring:

    def test_nominal_point_is_normal(self, trained_detector1):
        score, status = trained_detector1.score(65.0, 0.05, 1480.0)
        assert score  > THRESHOLD_WARNING, f"Nominal score {score:.4f} ≤ 0.0"
        assert status == "NORMAL"

    def test_near_nominal_is_normal(self, trained_detector1):
        _, status = trained_detector1.score(65.5, 0.051, 1481.0)
        assert status == "NORMAL"

    def test_score_is_float(self, trained_detector1):
        score, _ = trained_detector1.score(65.0, 0.05, 1480.0)
        assert isinstance(score, float)

    def test_score_range_on_normal_population(self, trained_detector1):
        """
        Over 100 normal samples, scores must mostly be positive.
        contamination=0.05 means at most ~5% false positives on training dist.
        Paper Table III reports Normal score range [−0.0756, +0.1696].
        """
        scores, _, _, _, _ = _score_scenario(trained_detector1, "normal")
        positive = sum(1 for s in scores if s > THRESHOLD_WARNING)
        assert positive >= 90, (
            f"Only {positive}/100 normal samples scored NORMAL "
            f"(expected ≥ 90, contamination=0.05)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Status label logic
# ─────────────────────────────────────────────────────────────────────────────

class TestStatusLabelLogic:

    def test_threshold_values(self):
        assert THRESHOLD_WARNING  == 0.0
        assert THRESHOLD_CRITICAL == -0.10

    def test_fault_names_map(self):
        assert FAULT_NAMES == {
            0: "NORMAL",
            1: "BEARING_FAULT",
            2: "STATOR_FAULT",
            3: "ROTOR_BAR_FAULT",
        }

    def test_normal_region(self, trained_detector1):
        score, status = trained_detector1.score(65.0, 0.05, 1480.0)
        assert score > THRESHOLD_WARNING
        assert status == "NORMAL"

    def test_bearing_fault_not_normal(self, trained_detector1):
        """Extreme bearing fault values must leave the NORMAL region."""
        score, status = trained_detector1.score(80.0, 0.40, 1470.0)
        assert score <= THRESHOLD_WARNING, (
            f"Bearing fault score {score:.4f} ≤ THRESHOLD_WARNING (0.0)")

    def test_stator_fault_not_normal(self, trained_detector1):
        score, status = trained_detector1.score(90.0, 0.12, 1460.0)
        assert score <= THRESHOLD_WARNING


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fault detection rates — core Table III result
# ─────────────────────────────────────────────────────────────────────────────

class TestFaultDetectionRates:
    """
    For each fault: 100 samples scored → detection rate ≥ 80%.
    Critical rate ≥ 60% for severe faults (bearing, stator).
    Paper result: all three faults → 100% CRITICAL (Table III).

    WHY 80% minimum?
    Isolation Forest is unsupervised — trained only on normal data.
    80% reflects conservative worst-case industrial performance.
    The actual result (100%) demonstrates strong fault separability
    under the evaluated conditions.
    """
    MIN_DETECTION = 0.80
    MIN_CRITICAL  = 0.60

    def test_bearing_detection_rate(self, trained_detector1):
        scores, _, _, _, _ = _score_scenario(trained_detector1, "bearing_fault")
        detected = sum(1 for s in scores if s < THRESHOLD_WARNING)
        assert detected / N_SAMPLES >= self.MIN_DETECTION, (
            f"Bearing detection {detected}/{N_SAMPLES} < {self.MIN_DETECTION:.0%}")

    def test_stator_detection_rate(self, trained_detector1):
        scores, _, _, _, _ = _score_scenario(trained_detector1, "stator_fault")
        detected = sum(1 for s in scores if s < THRESHOLD_WARNING)
        assert detected / N_SAMPLES >= self.MIN_DETECTION, (
            f"Stator detection {detected}/{N_SAMPLES} < {self.MIN_DETECTION:.0%}")

    def test_rotor_detection_rate(self, trained_detector1):
        scores, _, _, _, _ = _score_scenario(trained_detector1, "rotor_bar_fault")
        detected = sum(1 for s in scores if s < THRESHOLD_WARNING)
        assert detected / N_SAMPLES >= self.MIN_DETECTION, (
            f"Rotor detection {detected}/{N_SAMPLES} < {self.MIN_DETECTION:.0%}")

    def test_bearing_critical_rate(self, trained_detector1):
        scores, _, _, _, _ = _score_scenario(trained_detector1, "bearing_fault")
        critical = sum(1 for s in scores if s < THRESHOLD_CRITICAL)
        assert critical / N_SAMPLES >= self.MIN_CRITICAL

    def test_stator_critical_rate(self, trained_detector1):
        scores, _, _, _, _ = _score_scenario(trained_detector1, "stator_fault")
        critical = sum(1 for s in scores if s < THRESHOLD_CRITICAL)
        assert critical / N_SAMPLES >= self.MIN_CRITICAL

    def test_false_positive_rate(self, trained_detector1):
        """
        Normal data FP rate must be ≤ 10%.
        contamination=0.05 → expect ≈3% FP on in-distribution data.
        Paper Table III: 3.0% FP rate.
        """
        scores, _, _, _, _ = _score_scenario(trained_detector1, "normal", seed=99)
        fp = sum(1 for s in scores if s < THRESHOLD_WARNING)
        assert fp / N_SAMPLES <= 0.10, (
            f"False positive rate {fp}/{N_SAMPLES} = {fp/N_SAMPLES:.1%} > 10%")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Detection accuracy table — research paper Table III
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectionAccuracyTable:
    """
    Prints the formatted detection accuracy table for the research paper.
    Now includes current channel statistics alongside ML results.

    Run with:  python -m pytest tests/test_ml.py::TestDetectionAccuracyTable -v -s
    Output goes directly into Table III of the manuscript.
    """

    def test_print_detection_accuracy_table(self, trained_detector1):
        """
        ╔══════════════════╦═════════╦══════════╦══════════╦══════════════╦═══════════════╗
        ║ Fault Type       ║ Samples ║ Detected ║ Critical ║ Score Range  ║  FP Rate      ║
        ╠══════════════════╬═════════╬══════════╬══════════╬══════════════╬═══════════════╣
        ...
        """
        fault_scenarios = [
            ("normal",          "Normal"),
            ("bearing_fault",   "Bearing Fault"),
            ("stator_fault",    "Stator Fault"),
            ("rotor_bar_fault", "Rotor Bar Fault"),
        ]

        results = {}
        for scenario, label in fault_scenarios:
            scores, statuses, ia_arr, ib_arr, ic_arr = _score_scenario(
                trained_detector1, scenario, seed=42)

            if scenario == "normal":
                fp = sum(1 for s in scores if s < THRESHOLD_WARNING)
                results[scenario] = {
                    "label"    : label,
                    "tp"       : 0,
                    "fp"       : fp,
                    "critical" : 0,
                    "score_min": min(scores),
                    "score_max": max(scores),
                    "ia_rms"   : float(np.sqrt(np.mean(ia_arr**2))),
                    "ib_rms"   : float(np.sqrt(np.mean(ib_arr**2))),
                    "ic_rms"   : float(np.sqrt(np.mean(ic_arr**2))),
                }
            else:
                tp       = sum(1 for s in scores if s < THRESHOLD_WARNING)
                critical = sum(1 for s in scores if s < THRESHOLD_CRITICAL)
                results[scenario] = {
                    "label"    : label,
                    "tp"       : tp,
                    "fp"       : 0,
                    "critical" : critical,
                    "score_min": min(scores),
                    "score_max": max(scores),
                    "ia_rms"   : float(np.sqrt(np.mean(ia_arr**2))),
                    "ib_rms"   : float(np.sqrt(np.mean(ib_arr**2))),
                    "ic_rms"   : float(np.sqrt(np.mean(ic_arr**2))),
                }

        print("\n")
        print("  TABLE III — Detection Performance by Scenario (100 samples each)")
        print("  ╔══════════════════╦═════════╦══════════╦══════════╦══════════════════════════╗")
        print("  ║ Scenario         ║ Detect  ║ Critical ║ FP Rate  ║ Score Range              ║")
        print("  ╠══════════════════╬═════════╬══════════╬══════════╬══════════════════════════╣")
        for sc, r in results.items():
            if sc == "normal":
                det_str = "  —   "
                fp_str  = f"{r['fp']:3d}/100  ({r['fp']/N_SAMPLES*100:.1f}%)"
                crit_str= "  —   "
            else:
                det_str = f"{r['tp']:3d}/100"
                fp_str  = "  —       "
                crit_str= f"{r['critical']:3d}/100"
            print(f"  ║ {r['label']:<16} ║ {det_str:<7} ║ {crit_str:<8} ║ {fp_str:<8} "
                  f"║ [{r['score_min']:+.4f}, {r['score_max']:+.4f}]     ║")
        print("  ╚══════════════════╩═════════╩══════════╩══════════╩══════════════════════════╝")

        print("\n  Three-Phase Current RMS by Scenario (A):")
        print("  ╔══════════════════╦══════════╦══════════╦══════════╗")
        print("  ║ Scenario         ║  Ia RMS  ║  Ib RMS  ║  Ic RMS  ║")
        print("  ╠══════════════════╬══════════╬══════════╬══════════╣")
        for sc, r in results.items():
            print(f"  ║ {r['label']:<16} ║ {r['ia_rms']:>8.3f}  ║ "
                  f"{r['ib_rms']:>8.3f}  ║ {r['ic_rms']:>8.3f}  ║")
        print("  ╚══════════════════╩══════════╩══════════╩══════════╝")
        print()

        # Minimal assertions — all fault detection rates must be positive
        for sc in ("bearing_fault", "stator_fault", "rotor_bar_fault"):
            assert results[sc]["tp"] > 0, f"Zero detections for {sc}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature ablation — paper Table V
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureAblation:
    """
    Retrain detector using only one feature channel at a time.
    Validates Table V of the paper:
      - Temperature alone:  100% detection for bearing, stator, rotor
      - Vibration alone:    100% bearing, 0% stator, 0% rotor
      - Speed alone:        100% detection for all three faults
    Each detector is trained from scratch with one-dimensional features.
    """

    def _train_single_feature(self, feature_idx: int, motor_id="motor1"):
        """
        Train a 1-D Isolation Forest on one feature channel.
        feature_idx: 0=temp, 1=vib, 2=speed
        """
        from sklearn.ensemble import IsolationForest
        import numpy as np
        from sensor_simulator import generate_scenario

        df = generate_scenario("normal", duration_s=0.2, fs=1000, seed=42)
        cols = ["temperature_C", "vibration_x_g", "speed_rpm"]
        X = df[cols[feature_idx]].values[:200].reshape(-1, 1)
        model = IsolationForest(n_estimators=100, contamination=0.05,
                                random_state=42)
        model.fit(X)
        return model, cols[feature_idx]

    def _detection_rate_1d(self, model, col_name, scenario, n=100, seed=42):
        duration = n / 1000.0 + 0.1
        df = generate_scenario(scenario, duration_s=duration, fs=1000, seed=seed)
        df = df.head(n)
        X = df[col_name].values.reshape(-1, 1)
        scores = model.decision_function(X)
        detected = int(np.sum(scores < 0))
        return detected

    def test_temperature_only_detects_all_faults(self):
        model, col = self._train_single_feature(0)  # temperature
        for fault in ("bearing_fault", "stator_fault", "rotor_bar_fault"):
            detected = self._detection_rate_1d(model, col, fault)
            assert detected >= 80, (
                f"Temperature-only missed {fault}: {detected}/100 detected")

    def test_vibration_only_detects_bearing(self):
        """Vibration catches bearing fault (large RMS increase)."""
        model, col = self._train_single_feature(1)  # vibration
        detected = self._detection_rate_1d(model, col, "bearing_fault")
        assert detected >= 80, f"Vibration-only missed bearing: {detected}/100"

    def test_vibration_detects_bearing_best(self):
        """
        Vibration's discriminative power is strongest for bearing fault
        (6.7× RMS increase → high detection rate) and weakest for rotor bar
        fault (1.8× increase) and stator fault (2.4× increase).
        Bearing detection rate must be strictly higher than rotor bar rate —
        this reflects the fault severity hierarchy in vibration space.
        Paper Table V: bearing 100%, stator/rotor partial.
        """
        model, col = self._train_single_feature(1)
        det_bearing = self._detection_rate_1d(model, col, "bearing_fault")
        det_rotor   = self._detection_rate_1d(model, col, "rotor_bar_fault")
        assert det_bearing > det_rotor, (
            f"Bearing detection ({det_bearing}) should exceed rotor bar "
            f"({det_rotor}) for vibration-only model")

    def test_speed_only_detects_all_faults(self):
        model, col = self._train_single_feature(2)  # speed
        for fault in ("bearing_fault", "stator_fault", "rotor_bar_fault"):
            detected = self._detection_rate_1d(model, col, fault)
            assert detected >= 80, (
                f"Speed-only missed {fault}: {detected}/100 detected")

    def test_print_ablation_table(self):
        """Prints Table V for the manuscript."""
        from sklearn.ensemble import IsolationForest
        import numpy as np
        from sensor_simulator import generate_scenario

        features = ["temperature_C", "vibration_x_g", "speed_rpm"]
        feat_labels = ["Temperature only", "Vibration only", "Speed only"]
        faults = ["bearing_fault", "stator_fault", "rotor_bar_fault"]
        fault_labels = ["Bearing", "Stator", "Rotor"]

        # Train normal data
        df_normal = generate_scenario("normal", duration_s=0.2, fs=1000, seed=42)

        print("\n")
        print("  TABLE V — Feature Ablation (Motor 1, 100 independent samples/scenario)")
        print("  ╔══════════════════════╦══════════╦════════╦════════╦════════╗")
        print("  ║ Feature(s)           ║  FP Rate ║Bearing ║ Stator ║ Rotor  ║")
        print("  ╠══════════════════════╬══════════╬════════╬════════╬════════╣")

        for i, (feat, flabel) in enumerate(zip(features, feat_labels)):
            X_train = df_normal[feat].values[:200].reshape(-1, 1)
            model = IsolationForest(n_estimators=100, contamination=0.05,
                                    random_state=42)
            model.fit(X_train)

            # FP rate on normal
            df_test_n = generate_scenario("normal", duration_s=0.11, fs=1000, seed=99)
            X_n = df_test_n[feat].values[:100].reshape(-1, 1)
            fp = int(np.sum(model.decision_function(X_n) < 0))

            # Detection rates
            det = []
            for fault in faults:
                df_f = generate_scenario(fault, duration_s=0.11, fs=1000, seed=42)
                X_f  = df_f[feat].values[:100].reshape(-1, 1)
                d    = int(np.sum(model.decision_function(X_f) < 0))
                det.append(d)

            print(f"  ║ {flabel:<20} ║   {fp:3d}%   ║  {det[0]:3d}%  ║  {det[1]:3d}%  ║  {det[2]:3d}%  ║")

        # All three features (deployed)
        X_all = df_normal[features].values[:200]
        model_all = IsolationForest(n_estimators=100, contamination=0.05,
                                    random_state=42)
        model_all.fit(X_all)
        df_test_n = generate_scenario("normal", duration_s=0.11, fs=1000, seed=99)
        fp_all = int(np.sum(
            model_all.decision_function(df_test_n[features].values[:100]) < 0))
        det_all = []
        for fault in faults:
            df_f = generate_scenario(fault, duration_s=0.11, fs=1000, seed=42)
            d    = int(np.sum(
                model_all.decision_function(df_f[features].values[:100]) < 0))
            det_all.append(d)

        print(f"  ║ {'All three (deployed)':<20} ║   {fp_all:3d}%   ║  {det_all[0]:3d}%  "
              f"║  {det_all[1]:3d}%  ║  {det_all[2]:3d}%  ║")
        print("  ╚══════════════════════╩══════════╩════════╩════════╩════════╝")
        print()
