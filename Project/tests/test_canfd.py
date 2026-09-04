"""
test_canfd.py — CAN FD Frame Integrity Tests (v2 — 32-byte PRIMARY frame)
=========================================================================
Updated for the upgraded PRIMARY frame layout:

  OLD: '<fffBxxx'         = 16 bytes  (3 floats + uint8 + 3 pad)
  NEW: '<ffffffBxxxxxxx'  = 32 bytes  (6 floats + uint8 + 7 pad)
        └─ NEW: current_a_A, current_b_A, current_c_A

_unpack_primary() now returns 7 values:
  (temp, vib, variable, current_a, current_b, current_c, fault_id)

build_primary_frame() now takes 3 extra current arguments:
  build_primary_frame(arb_id, temp, vib, variable,
                      current_a, current_b, current_c, fault_id)

WHY float32 round-trip tolerance = 0.01?
  struct '<f' packs Python float64 → IEEE-754 float32 (4 bytes).
  Float32 has ~7 significant decimal digits. For values in our range:
    temp=65°C      → error ≈ 65 × 2^-23 ≈ 7.7×10^-6  (<0.01)
    vib=0.05g      → error ≈ 6×10^-9                  (<0.01)
    speed=1480 RPM → error ≈ 1.8×10^-4                (<0.01)
    current=39.6 A → error ≈ 4.7×10^-6                (<0.01)
  0.01 safely covers all values in our operating range.

WHY len(msg.data) not msg.dlc?
  python-can dlc is the CAN DLC code (0-15), which maps non-linearly
  for DLC > 8 in CAN FD: DLC=10 → 16 bytes, DLC=12 → 20 bytes.
  len(msg.data) gives the actual byte count — unambiguous.

Industrial significance (FAT Level 2 — Communication Interface Test):
  In real deployment an integrator verifies frame byte-order and field
  mapping on an oscilloscope/CAN analyser. These struct round-trip tests
  are the software equivalent — providing the same guarantee automatically.
"""

import struct
import pytest

from can_node import (
    build_primary_frame,
    build_secondary_frame,
    PRIMARY_FMT,
    SECONDARY_FMT,
    FAULT_ID_MAP,
    FRAME_MOTOR1_PRIMARY,
    FRAME_MOTOR1_SECONDARY,
    FRAME_MOTOR2_PRIMARY,
    FRAME_MOTOR2_SECONDARY,
)
from ml_gateway import _unpack_primary, _unpack_secondary

FLOAT_TOL = 0.01   # IEEE-754 float32 quantisation tolerance

# Typical 3-phase currents for a healthy 28 A rated motor
# Balanced: Ia + Ib + Ic ≈ 0 (phasor sum)
IA_NOM  =  39.2   # ≈ 28 × √2
IB_NOM  = -19.6   # ≈ -IA/2 (120° phase)
IC_NOM  = -19.6   # ≈ -IA/2 (240° phase)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Struct format sizes
# ─────────────────────────────────────────────────────────────────────────────

class TestStructFormats:
    """
    The PRIMARY frame is 32 bytes — a valid CAN FD DLC.
    Valid CAN FD DLC byte counts: 0-8, 12, 16, 20, 24, 32, 48, 64.
    32 bytes is required because 6 float32 + 1 uint8 + 7 pad = 32.
    The old 16-byte frame only had 3 floats (no current channels).
    """

    def test_primary_fmt_size_is_32(self):
        """UPGRADED: PRIMARY frame is now 32 bytes (was 16)."""
        size = struct.calcsize(PRIMARY_FMT)
        assert size == 32, (
            f"PRIMARY_FMT '{PRIMARY_FMT}' → {size} bytes, expected 32.\n"
            f"  6×float32 (24 B) + 1×uint8 (1 B) + 7×pad (7 B) = 32 B")

    def test_secondary_fmt_size_is_8(self):
        assert struct.calcsize(SECONDARY_FMT) == 8

    def test_primary_is_little_endian(self):
        assert PRIMARY_FMT.startswith("<"), f"PRIMARY_FMT not little-endian: '{PRIMARY_FMT}'"

    def test_secondary_is_little_endian(self):
        assert SECONDARY_FMT.startswith("<")

    def test_primary_contains_six_floats(self):
        """Count 'f' characters — must be exactly 6 for the 6 sensor channels."""
        n_floats = PRIMARY_FMT.count('f')
        assert n_floats == 6, (
            f"PRIMARY_FMT has {n_floats} float fields, expected 6 "
            f"(temp, vib, speed/flow, Ia, Ib, Ic)")

    def test_primary_contains_uint8(self):
        """'B' is the fault_id byte."""
        assert 'B' in PRIMARY_FMT

    def test_unpack_returns_7_values(self):
        """
        UPGRADED: _unpack_primary now returns 7 values:
        (temp, vib, variable, current_a, current_b, current_c, fault_id)
        The old version returned 4 values — callers must be updated.
        """
        msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0,
                                  IA_NOM, IB_NOM, IC_NOM, 0)
        result = _unpack_primary(msg.data)
        assert len(result) == 7, (
            f"_unpack_primary returned {len(result)} values, expected 7: "
            f"(temp, vib, variable, Ia, Ib, Ic, fault_id)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Frame flags — is_fd and is_extended_id
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameFlags:
    """
    WHY is_fd must be True?
      PRIMARY frame is 32 bytes — exceeds the 8-byte classic CAN limit.
      is_fd=True sets the FDF bit so CAN FD transceivers accept the frame.

    WHY is_extended_id must be False?
      We use standard 11-bit arbitration IDs (0x000–0x7FF).
      Extended 29-bit IDs add 18 bits of overhead — unnecessary for 4 IDs.
    """

    FRAME_CASES = [
        (FRAME_MOTOR1_PRIMARY,   0x100, "primary",   "motor1"),
        (FRAME_MOTOR1_SECONDARY, 0x101, "secondary", "motor1"),
        (FRAME_MOTOR2_PRIMARY,   0x200, "primary",   "motor2"),
        (FRAME_MOTOR2_SECONDARY, 0x201, "secondary", "motor2"),
    ]

    def _build(self, arb_id, ftype):
        if ftype == "primary":
            return build_primary_frame(arb_id, 65.0, 0.05, 1480.0,
                                       IA_NOM, IB_NOM, IC_NOM, 0)
        return build_secondary_frame(arb_id, 27.0, 50.0)

    @pytest.mark.parametrize("arb_id,expected_id,ftype,motor", FRAME_CASES)
    def test_is_fd_flag(self, arb_id, expected_id, ftype, motor):
        msg = self._build(arb_id, ftype)
        assert msg.is_fd is True, (
            f"Frame 0x{arb_id:03X} ({motor} {ftype}) is_fd=False")

    @pytest.mark.parametrize("arb_id,expected_id,ftype,motor", FRAME_CASES)
    def test_is_extended_id_false(self, arb_id, expected_id, ftype, motor):
        msg = self._build(arb_id, ftype)
        assert msg.is_extended_id is False, (
            f"Frame 0x{arb_id:03X} is_extended_id=True (should be 11-bit standard)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Arbitration IDs
# ─────────────────────────────────────────────────────────────────────────────

class TestArbitrationIDs:

    def test_motor1_primary_id(self):
        msg = build_primary_frame(FRAME_MOTOR1_PRIMARY, 65.0, 0.05, 1480.0,
                                  IA_NOM, IB_NOM, IC_NOM, 0)
        assert msg.arbitration_id == 0x100

    def test_motor1_secondary_id(self):
        msg = build_secondary_frame(FRAME_MOTOR1_SECONDARY, 27.0, 50.0)
        assert msg.arbitration_id == 0x101

    def test_motor2_primary_id(self):
        msg = build_primary_frame(FRAME_MOTOR2_PRIMARY, 63.0, 0.05, 100.0,
                                  IA_NOM*0.97, IB_NOM*0.97, IC_NOM*0.97, 0)
        assert msg.arbitration_id == 0x200

    def test_motor2_secondary_id(self):
        msg = build_secondary_frame(FRAME_MOTOR2_SECONDARY, 27.5, 49.0)
        assert msg.arbitration_id == 0x201

    def test_custom_id_passthrough(self):
        for arb_id in [0x000, 0x100, 0x200, 0x7FF]:
            msg = build_primary_frame(arb_id, 65.0, 0.05, 1480.0,
                                      IA_NOM, IB_NOM, IC_NOM, 0)
            assert msg.arbitration_id == arb_id


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data length
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLength:
    """
    PRIMARY  = 32 bytes (UPGRADED from 16)
    SECONDARY = 8 bytes (unchanged)
    """

    def test_primary_frame_length_32(self):
        """UPGRADED: primary frames are now 32 bytes, not 16."""
        for arb_id in [FRAME_MOTOR1_PRIMARY, FRAME_MOTOR2_PRIMARY]:
            msg = build_primary_frame(arb_id, 65.0, 0.05, 1480.0,
                                      IA_NOM, IB_NOM, IC_NOM, 0)
            assert len(msg.data) == 32, (
                f"Frame 0x{arb_id:03X} data length {len(msg.data)}, expected 32")

    def test_secondary_frame_length_8(self):
        for arb_id in [FRAME_MOTOR1_SECONDARY, FRAME_MOTOR2_SECONDARY]:
            msg = build_secondary_frame(arb_id, 27.0, 50.0)
            assert len(msg.data) == 8, (
                f"Frame 0x{arb_id:03X} data length {len(msg.data)}, expected 8")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRIMARY round-trip — all 7 fields including 3-phase currents
# ─────────────────────────────────────────────────────────────────────────────

# (arb_id, temp, vib, variable, ia, ib, ic, fault_id)
PRIMARY_CASES = [
    # Normal Motor 1
    (0x100, 65.0,  0.05,   1480.0,  39.20,  -19.60, -19.60, 0),
    # Bearing fault Motor 1
    (0x100, 80.3,  0.342,  1470.1,  39.20,  -19.60, -19.60, 1),
    # Stator fault Motor 1 — Phase-A imbalanced
    (0x100, 88.2,  0.12,   1460.0,  44.50,  -19.20, -19.20, 2),
    # Rotor bar fault Motor 1 — sidebands in all phases
    (0x100, 73.1,  0.08,   1465.0,  41.10,  -20.55, -20.55, 3),
    # Normal Motor 2 (pump — uses flow_rate not rpm)
    (0x200, 63.05, 0.0525, 100.0,   38.02,  -19.01, -19.01, 0),
    # Bearing fault Motor 2
    (0x200, 77.9,  0.359,  97.0,    38.02,  -19.01, -19.01, 1),
]

class TestPrimaryRoundTrip:
    """
    Full round-trip: build_primary_frame() → _unpack_primary() → compare.
    All 7 fields verified: temp, vib, variable, Ia, Ib, Ic, fault_id.
    """

    @pytest.mark.parametrize("arb_id,temp,vib,var,ia,ib,ic,fid", PRIMARY_CASES)
    def test_temperature(self, arb_id, temp, vib, var, ia, ib, ic, fid):
        msg = build_primary_frame(arb_id, temp, vib, var, ia, ib, ic, fid)
        t_out, *_ = _unpack_primary(msg.data)
        assert abs(t_out - temp) < FLOAT_TOL

    @pytest.mark.parametrize("arb_id,temp,vib,var,ia,ib,ic,fid", PRIMARY_CASES)
    def test_vibration(self, arb_id, temp, vib, var, ia, ib, ic, fid):
        msg = build_primary_frame(arb_id, temp, vib, var, ia, ib, ic, fid)
        _, v_out, *_ = _unpack_primary(msg.data)
        assert abs(v_out - vib) < FLOAT_TOL

    @pytest.mark.parametrize("arb_id,temp,vib,var,ia,ib,ic,fid", PRIMARY_CASES)
    def test_variable(self, arb_id, temp, vib, var, ia, ib, ic, fid):
        msg = build_primary_frame(arb_id, temp, vib, var, ia, ib, ic, fid)
        _, _, var_out, *_ = _unpack_primary(msg.data)
        assert abs(var_out - var) < FLOAT_TOL

    @pytest.mark.parametrize("arb_id,temp,vib,var,ia,ib,ic,fid", PRIMARY_CASES)
    def test_current_a(self, arb_id, temp, vib, var, ia, ib, ic, fid):
        """UPGRADED: Phase-A current round-trip."""
        msg = build_primary_frame(arb_id, temp, vib, var, ia, ib, ic, fid)
        _, _, _, ia_out, _, _, _ = _unpack_primary(msg.data)
        assert abs(ia_out - ia) < FLOAT_TOL, (
            f"Phase-A round-trip error: packed {ia}, got {ia_out}")

    @pytest.mark.parametrize("arb_id,temp,vib,var,ia,ib,ic,fid", PRIMARY_CASES)
    def test_current_b(self, arb_id, temp, vib, var, ia, ib, ic, fid):
        """UPGRADED: Phase-B current round-trip."""
        msg = build_primary_frame(arb_id, temp, vib, var, ia, ib, ic, fid)
        _, _, _, _, ib_out, _, _ = _unpack_primary(msg.data)
        assert abs(ib_out - ib) < FLOAT_TOL, (
            f"Phase-B round-trip error: packed {ib}, got {ib_out}")

    @pytest.mark.parametrize("arb_id,temp,vib,var,ia,ib,ic,fid", PRIMARY_CASES)
    def test_current_c(self, arb_id, temp, vib, var, ia, ib, ic, fid):
        """UPGRADED: Phase-C current round-trip."""
        msg = build_primary_frame(arb_id, temp, vib, var, ia, ib, ic, fid)
        _, _, _, _, _, ic_out, _ = _unpack_primary(msg.data)
        assert abs(ic_out - ic) < FLOAT_TOL, (
            f"Phase-C round-trip error: packed {ic}, got {ic_out}")

    @pytest.mark.parametrize("arb_id,temp,vib,var,ia,ib,ic,fid", PRIMARY_CASES)
    def test_fault_id_exact(self, arb_id, temp, vib, var, ia, ib, ic, fid):
        """fault_id is uint8 — no float quantisation, must be exact."""
        msg = build_primary_frame(arb_id, temp, vib, var, ia, ib, ic, fid)
        *_, fid_out = _unpack_primary(msg.data)
        assert fid_out == fid


# ─────────────────────────────────────────────────────────────────────────────
# 6. SECONDARY round-trip
# ─────────────────────────────────────────────────────────────────────────────

SECONDARY_CASES = [
    (0x101, 27.0,  50.0),
    (0x101, 10.0,  30.0),
    (0x101, 45.0,  80.0),
    (0x201, 27.5,  49.0),
    (0x201, 15.3,  62.5),
]

class TestSecondaryRoundTrip:

    @pytest.mark.parametrize("arb_id,amb,hum", SECONDARY_CASES)
    def test_ambient_temp(self, arb_id, amb, hum):
        msg = build_secondary_frame(arb_id, amb, hum)
        amb_out, _ = _unpack_secondary(msg.data)
        assert abs(amb_out - amb) < FLOAT_TOL

    @pytest.mark.parametrize("arb_id,amb,hum", SECONDARY_CASES)
    def test_humidity(self, arb_id, amb, hum):
        msg = build_secondary_frame(arb_id, amb, hum)
        _, hum_out = _unpack_secondary(msg.data)
        assert abs(hum_out - hum) < FLOAT_TOL


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fault ID encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestFaultIDEncoding:

    @pytest.mark.parametrize("scenario,expected", [
        ("normal", 0), ("bearing_fault", 1),
        ("stator_fault", 2), ("rotor_bar_fault", 3),
    ])
    def test_fault_id_map(self, scenario, expected):
        assert FAULT_ID_MAP[scenario] == expected

    @pytest.mark.parametrize("scenario,expected", [
        ("normal", 0), ("bearing_fault", 1),
        ("stator_fault", 2), ("rotor_bar_fault", 3),
    ])
    def test_fault_id_round_trip(self, scenario, expected):
        fid = FAULT_ID_MAP[scenario]
        msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0,
                                  IA_NOM, IB_NOM, IC_NOM, fid)
        *_, fid_out = _unpack_primary(msg.data)
        assert fid_out == expected

    def test_fault_id_boundary_255(self):
        msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0,
                                  IA_NOM, IB_NOM, IC_NOM, 255)
        *_, fid_out = _unpack_primary(msg.data)
        assert fid_out == 255

    def test_fault_id_boundary_0(self):
        msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0,
                                  IA_NOM, IB_NOM, IC_NOM, 0)
        *_, fid_out = _unpack_primary(msg.data)
        assert fid_out == 0

    def test_three_phase_currents_in_frame_for_all_faults(self):
        """
        All fault scenarios must pack valid (non-zero) current values.
        The 3-phase currents are now mandatory fields in the PRIMARY frame.
        """
        for scenario in ("normal", "bearing_fault", "stator_fault", "rotor_bar_fault"):
            fid = FAULT_ID_MAP[scenario]
            msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0,
                                      IA_NOM, IB_NOM, IC_NOM, fid)
            _, _, _, ia, ib, ic, _ = _unpack_primary(msg.data)
            assert ia != 0.0, f"Phase-A is zero for {scenario}"
            assert ib != 0.0, f"Phase-B is zero for {scenario}"
            assert ic != 0.0, f"Phase-C is zero for {scenario}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Error handling — truncated data
# ─────────────────────────────────────────────────────────────────────────────

class TestUnpackErrorHandling:

    def test_primary_too_short_raises(self):
        """UPGRADED: PRIMARY_SIZE is now 32 — data < 32 must raise."""
        with pytest.raises(ValueError, match="too short"):
            _unpack_primary(bytes(16))   # old size — now too short

    def test_secondary_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            _unpack_secondary(bytes(4))

    def test_primary_accepts_exact_32_bytes(self):
        msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0,
                                  IA_NOM, IB_NOM, IC_NOM, 0)
        result = _unpack_primary(bytes(msg.data))
        assert len(result) == 7

    def test_secondary_accepts_exact_8_bytes(self):
        msg = build_secondary_frame(0x101, 27.0, 50.0)
        result = _unpack_secondary(bytes(msg.data))
        assert len(result) == 2

    def test_primary_accepts_longer_data(self):
        """unpack_from silently ignores trailing bytes — future-proof."""
        result = _unpack_primary(bytes(48))
        assert len(result) == 7

    def test_16_bytes_now_too_short(self):
        """
        Regression guard: the OLD 16-byte frame size must now raise
        ValueError — confirms the upgrade is enforced, not silently ignored.
        """
        with pytest.raises(ValueError, match="too short"):
            _unpack_primary(bytes(16))
