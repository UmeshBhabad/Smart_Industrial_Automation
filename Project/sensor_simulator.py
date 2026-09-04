import numpy as np
import pandas as pd
from typing import Literal

# ---------------------------------------------------------------------------
# Motor nameplate / base parameters
# ---------------------------------------------------------------------------
MOTOR = dict(
    rated_power_kW   = 15,
    rated_voltage_V  = 415,       # line-to-line, 50 Hz
    rated_current_A  = 28.0,
    rated_speed_rpm  = 1480,      # ~2% slip on 4-pole, 50 Hz
    poles            = 4,
    supply_freq_Hz   = 50,
    bearing_bpfo     = 157.0,     # Ball-Pass Freq Outer race (Hz) — typical
    bearing_bpfi     = 218.0,     # Ball-Pass Freq Inner race (Hz)
    bearing_bsf      = 68.5,      # Ball Spin Frequency (Hz)
    rotor_bars       = 28,
)

SCENARIOS = ("normal", "bearing_fault", "stator_fault", "rotor_bar_fault")

ScenarioLabel = Literal["normal", "bearing_fault", "stator_fault", "rotor_bar_fault"]

# ---------------------------------------------------------------------------
# Random-seed helper
# ---------------------------------------------------------------------------
RNG_SEED = 42

def _rng(seed=RNG_SEED):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_scenario(
    scenario: ScenarioLabel = "normal",
    duration_s: float = 5.0,
    fs: float = 5000.0,        # samples per second
    seed: int = RNG_SEED,
    add_noise: bool = True,
) -> pd.DataFrame:
    """
    Simulate sensor readings for the chosen Motor-1 operating scenario.

    Parameters
    ----------
    scenario    : one of SCENARIOS
    duration_s  : total simulation time in seconds
    fs          : sampling frequency (Hz)
    seed        : random seed for reproducibility
    add_noise   : add realistic sensor noise floor

    Returns
    -------
    pandas DataFrame with columns listed in module docstring.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got '{scenario}'")

    rng   = _rng(seed)
    N     = int(duration_s * fs)
    t     = np.linspace(0, duration_s, N, endpoint=False)
    ws    = 2 * np.pi * MOTOR["supply_freq_Hz"]   # supply angular freq

    # ------------------------------------------------------------------
    # Base (healthy) signals
    # ------------------------------------------------------------------
    # Phase currents  (balanced three-phase)
    Ia = MOTOR["rated_current_A"] * np.sqrt(2) * np.sin(ws * t)
    Ib = MOTOR["rated_current_A"] * np.sqrt(2) * np.sin(ws * t - 2 * np.pi / 3)
    Ic = MOTOR["rated_current_A"] * np.sqrt(2) * np.sin(ws * t + 2 * np.pi / 3)

    # Shaft speed — slight slip ripple
    speed = MOTOR["rated_speed_rpm"] + 2.0 * np.sin(2 * np.pi * 0.5 * t)

    # Vibration — mechanical unbalance (1× running speed) + harmonics
    fr     = MOTOR["rated_speed_rpm"] / 60.0          # rotational freq
    vib_x  = (0.05 * np.sin(2 * np.pi * fr * t) +
               0.02 * np.sin(2 * np.pi * 2 * fr * t))
    vib_y  = (0.05 * np.cos(2 * np.pi * fr * t) +
               0.02 * np.cos(2 * np.pi * 2 * fr * t))
    vib_z  =  0.01 * np.sin(2 * np.pi * fr * t)

    # Temperature — ambient + steady-state rise
    temp   = 65.0 + 2.0 * np.sin(2 * np.pi * 0.01 * t)

    # Sound
    sound  = 72.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)

    # ------------------------------------------------------------------
    # Fault overlays
    # ------------------------------------------------------------------
    if scenario == "bearing_fault":
        # Outer-race defect: impulsive burst at BPFO, amplitude modulated
        # by shaft rotation → sidebands at fr around BPFO
        bpfo  = MOTOR["bearing_bpfo"]
        # Impulsive carrier at BPFO with 1× AM
        carrier = (1 + 0.4 * np.sin(2 * np.pi * fr * t)) * np.sin(2 * np.pi * bpfo * t)
        vib_x  += 0.30 * carrier
        vib_y  += 0.25 * carrier
        vib_z  += 0.10 * carrier

        # Inner-race component (sidebands at fr around BPFI)
        bpfi   = MOTOR["bearing_bpfi"]
        carrier_i = (1 + 0.3 * np.sin(2 * np.pi * fr * t)) * np.sin(2 * np.pi * bpfi * t)
        vib_x  += 0.15 * carrier_i

        # Thermal effect — bearing friction raises temp
        temp  += 15.0 + 5.0 * np.sin(2 * np.pi * 0.005 * t)

        # Speed drops slightly due to friction
        speed -= 10.0

        # Acoustic
        sound += 8.0

    elif scenario == "stator_fault":
        # Turn-to-turn short: introduces negative sequence → unbalanced currents
        # and odd harmonics (3rd, 5th) in the faulted phase
        fault_depth = 0.12   # 12 % winding shorted

        # Unbalance: phase-A gets additional 3rd harmonic and DC offset
        Ia += (fault_depth * MOTOR["rated_current_A"] * np.sqrt(2) *
               np.sin(3 * ws * t + 0.3))
        Ia += fault_depth * 4.0   # DC offset from circulating current

        # 5th harmonic pollution on all phases
        h5 = 0.06 * MOTOR["rated_current_A"] * np.sqrt(2)
        Ia += h5 * np.sin(5 * ws * t)
        Ib += h5 * np.sin(5 * ws * t - 2 * np.pi / 3)
        Ic += h5 * np.sin(5 * ws * t + 2 * np.pi / 3)

        # Thermal — shorted turns create I²R hotspot
        temp  += 22.0 + 8.0 * np.sin(2 * np.pi * 0.008 * t)

        # Torque ripple → vibration at 2× supply (100 Hz) and sidebands
        vib_x += 0.12 * np.sin(2 * np.pi * 100 * t)
        vib_y += 0.10 * np.sin(2 * np.pi * 100 * t + np.pi / 4)

        # Speed ripple
        speed -= 20.0 + 5.0 * np.sin(2 * np.pi * 2 * fr * t)

        # Sound
        sound += 5.0

    elif scenario == "rotor_bar_fault":
        # Broken rotor bars: characteristic sidebands at (1±2ks)fs
        # where s = slip and k = 1, 2, 3 …
        slip     = (MOTOR["rated_speed_rpm"] - speed) / 1500.0   # ≈ 0.013
        # Scalar slip for sideband computation (use mean)
        s_mean   = 0.0133

        for k in (1, 2, 3):
            f_lower = MOTOR["supply_freq_Hz"] * (1 - 2 * k * s_mean)
            f_upper = MOTOR["supply_freq_Hz"] * (1 + 2 * k * s_mean)
            amp     = 0.04 / k   # decaying amplitude with order

            Ia += amp * MOTOR["rated_current_A"] * np.sqrt(2) * (
                np.sin(2 * np.pi * f_lower * t) + np.sin(2 * np.pi * f_upper * t)
            )
            Ib += amp * MOTOR["rated_current_A"] * np.sqrt(2) * (
                np.sin(2 * np.pi * f_lower * t - 2 * np.pi / 3) +
                np.sin(2 * np.pi * f_upper * t - 2 * np.pi / 3)
            )
            Ic += amp * MOTOR["rated_current_A"] * np.sqrt(2) * (
                np.sin(2 * np.pi * f_lower * t + 2 * np.pi / 3) +
                np.sin(2 * np.pi * f_upper * t + 2 * np.pi / 3)
            )

        # Rotor-bar pass frequency sidebands in vibration
        f_rbp = MOTOR["rotor_bars"] * fr    # ≈ 686 Hz for 28 bars
        vib_x += 0.08 * np.sin(2 * np.pi * f_rbp * t)
        vib_y += 0.07 * np.sin(2 * np.pi * f_rbp * t + np.pi / 6)

        # Speed fluctuation — torque pulsation per broken bar
        speed -= 15.0 + 8.0 * np.sin(2 * np.pi * fr * t)

        # Temperature — slightly elevated rotor losses
        temp  += 8.0

        # Sound
        sound += 4.0

    # ------------------------------------------------------------------
    # Noise floor
    # ------------------------------------------------------------------
    if add_noise:
        vib_x  += rng.normal(0, 0.005, N)
        vib_y  += rng.normal(0, 0.005, N)
        vib_z  += rng.normal(0, 0.003, N)
        Ia     += rng.normal(0, 0.20, N)
        Ib     += rng.normal(0, 0.20, N)
        Ic     += rng.normal(0, 0.20, N)
        temp   += rng.normal(0, 0.30, N)
        speed  += rng.normal(0, 0.50, N)
        sound  += rng.normal(0, 0.20, N)

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "time_s"        : t,
        "vibration_x_g" : vib_x,
        "vibration_y_g" : vib_y,
        "vibration_z_g" : vib_z,
        "current_a_A"   : Ia,
        "current_b_A"   : Ib,
        "current_c_A"   : Ic,
        "temperature_C" : temp,
        "speed_rpm"     : speed,
        "sound_dB"      : sound,
        "scenario"      : scenario,
    })
    return df


# ---------------------------------------------------------------------------
# Convenience: generate all four scenarios and save CSVs
# ---------------------------------------------------------------------------

def generate_all(
    duration_s: float = 5.0,
    fs: float = 5000.0,
    output_dir: str = ".",
    save_csv: bool = True,
) -> dict:
    """
    Generate all four scenarios and return them as a dict of DataFrames.
    Optionally save each to a CSV file in output_dir.

    Returns
    -------
    dict mapping scenario name → DataFrame
    """
    import os
    results = {}
    for sc in SCENARIOS:
        df = generate_scenario(sc, duration_s=duration_s, fs=fs)
        results[sc] = df
        if save_csv:
            path = os.path.join(output_dir, f"motor1_{sc}.csv")
            df.to_csv(path, index=False)
            print(f"  Saved {len(df):,} rows → {path}")
    return results


# ---------------------------------------------------------------------------
# Feature extractor (statistical + spectral — ready for ML pipeline)
# ---------------------------------------------------------------------------

def extract_features(df: pd.DataFrame, fs: float = 5000.0) -> dict:
    """
    Compute a flat feature dict from one scenario DataFrame.
    Useful for building a feature matrix for classification.

    Statistical features per channel:
        mean, std, rms, peak, crest_factor, kurtosis, skewness

    Spectral features (vibration_x only):
        dominant_freq_Hz, spectral_centroid_Hz
    """
    from scipy.stats import kurtosis, skew
    import scipy.fft as fft

    channels = [
        "vibration_x_g", "vibration_y_g", "vibration_z_g",
        "current_a_A", "current_b_A", "current_c_A",
        "temperature_C", "speed_rpm", "sound_dB",
    ]
    feats = {"scenario": df["scenario"].iloc[0]}

    for ch in channels:
        x  = df[ch].values
        rms = np.sqrt(np.mean(x ** 2))
        pk  = np.max(np.abs(x))
        feats[f"{ch}_mean"]         = float(np.mean(x))
        feats[f"{ch}_std"]          = float(np.std(x))
        feats[f"{ch}_rms"]          = float(rms)
        feats[f"{ch}_peak"]         = float(pk)
        feats[f"{ch}_crest_factor"] = float(pk / rms if rms > 0 else 0)
        feats[f"{ch}_kurtosis"]     = float(kurtosis(x))
        feats[f"{ch}_skewness"]     = float(skew(x))

    # Spectral — vibration X
    vx    = df["vibration_x_g"].values
    N     = len(vx)
    freqs = fft.rfftfreq(N, d=1.0 / fs)
    mag   = np.abs(fft.rfft(vx))
    feats["vib_x_dominant_freq_Hz"]    = float(freqs[np.argmax(mag)])
    feats["vib_x_spectral_centroid_Hz"] = float(
        np.sum(freqs * mag) / (np.sum(mag) + 1e-12)
    )
    return feats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os, json

    parser = argparse.ArgumentParser(
        description="Motor 1 sensor data simulator — induction motor"
    )
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS) + ["all"], default="all",
        help="Which scenario to generate (default: all)"
    )
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration in seconds (default: 5)")
    parser.add_argument("--fs", type=float, default=5000.0,
                        help="Sampling frequency in Hz (default: 5000)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save CSV files (default: current dir)")
    parser.add_argument("--features", action="store_true",
                        help="Also print extracted features as JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scenario == "all":
        results = generate_all(
            duration_s=args.duration, fs=args.fs,
            output_dir=args.output_dir, save_csv=True
        )
    else:
        df = generate_scenario(args.scenario, duration_s=args.duration, fs=args.fs)
        path = os.path.join(args.output_dir, f"motor1_{args.scenario}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {len(df):,} rows → {path}")
        results = {args.scenario: df}

    if args.features:
        for sc, df in results.items():
            feats = extract_features(df, fs=args.fs)
            print(f"\n--- Features: {sc} ---")
            print(json.dumps(feats, indent=2))