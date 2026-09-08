"""
analyze_angles.py
==================
Empirically determine the radar's actual angular resolution (spokes per
full rotation) from a real radar_raw .bin export, instead of assuming
a textbook figure like 4096 or 2048.

Reuses the exact header-parsing logic from Radar_Image_Generator.py so
results are consistent with what the renderer sees.

Usage:
    python analyze_angles.py sweep_1006.bin
"""

import struct
import sys
from pathlib import Path
from collections import Counter

import numpy as np

NAVICO_HEADER_LEN = 24
BYTES_PER_SPOKE   = 512
SPOKE_RECORD_LEN  = NAVICO_HEADER_LEN + BYTES_PER_SPOKE  # 536
FRAME_HDR_LEN     = 8
SPOKES_PER_FRAME  = 32
FRAME_LEN         = FRAME_HDR_LEN + SPOKES_PER_FRAME * SPOKE_RECORD_LEN  # 17160


def parse_spoke_header(data: bytes, offset: int = 0):
    """Same logic as Radar_Image_Generator.py parse_spoke_header, but we only
    need header_len/status validity + raw angle integer (not converted to degrees,
    since we don't yet know how many divisions = 360deg)."""
    if len(data) < offset + NAVICO_HEADER_LEN + BYTES_PER_SPOKE:
        return None

    header_len = data[offset]
    status = data[offset + 1]

    if header_len != NAVICO_HEADER_LEN:
        return None
    if status not in (0x02, 0x12):
        return None

    angle_raw = struct.unpack_from("<H", data, offset + 8)[0]
    return header_len, angle_raw


def read_all_spoke_angles(data: bytes):
    """Walk the file the same way read_spokes_navico_frames does, but just
    collect raw angle integers (and which frame/row they came from)."""
    angles = []
    offset = 0
    n = len(data)

    def spoke_valid(off):
        return parse_spoke_header(data, off) is not None

    while offset < n:
        spoke_start = offset
        if offset + FRAME_HDR_LEN + SPOKE_RECORD_LEN <= n and spoke_valid(offset + FRAME_HDR_LEN):
            spoke_start = offset + FRAME_HDR_LEN
        elif not spoke_valid(offset):
            offset += 1
            continue

        frame_angles = []
        for _ in range(SPOKES_PER_FRAME):
            if spoke_start + SPOKE_RECORD_LEN > n:
                break
            res = parse_spoke_header(data, spoke_start)
            if res is None:
                break
            _, angle_raw = res
            frame_angles.append(angle_raw)
            spoke_start += SPOKE_RECORD_LEN

        angles.append(frame_angles)

        if spoke_start <= offset:
            offset += 1
        else:
            offset = spoke_start

    return angles


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python analyze_angles.py <radar_raw_export.bin>")

    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"File not found: {path}")

    data = path.read_bytes()
    print(f"File: {path}  ({len(data):,} bytes)")

    frames = read_all_spoke_angles(data)
    all_angles = [a for frame in frames for a in frame]

    if not all_angles:
        sys.exit("No valid spokes parsed - check the file is a radar_raw export.")

    print(f"\nFrames parsed: {len(frames)}")
    print(f"Total spokes:  {len(all_angles)}")

    max_angle = max(all_angles)
    min_angle = min(all_angles)
    distinct = sorted(set(all_angles))

    print(f"\nRaw angle value range: {min_angle} .. {max_angle}")
    print(f"Distinct angle values seen: {len(distinct)}")

    # Diffs between consecutive distinct angles -> reveals the step size
    diffs = np.diff(distinct)
    if len(diffs) > 0:
        step_counter = Counter(diffs.tolist())
        common_step, count = step_counter.most_common(1)[0]
        print(f"Most common step between consecutive angle values: {common_step}  "
              f"(seen {count}/{len(diffs)} times)")
    else:
        common_step = None

    # The angle field is a fixed-width counter that wraps at some power-of-two
    # boundary (full circle = ANGLE_DIVISIONS ticks). We can't directly observe
    # ANGLE_DIVISIONS unless we see the wraparound, but we CAN bound it:
    #   ANGLE_DIVISIONS must be > max_angle observed
    # and if spokes are evenly spaced by `common_step`, then
    #   ANGLE_DIVISIONS is very likely a multiple of common_step.
    print(f"\n--- Inference ---")
    print(f"Any candidate ANGLE_DIVISIONS must be > max observed raw angle ({max_angle}).")
    for candidate in (1024, 2048, 4096, 8192):
        plausible = candidate > max_angle
        note = ""
        if plausible and common_step:
            note = "consistent" if candidate % common_step == 0 else "inconsistent w/ step size"
        print(f"  {candidate:5d}: {'possible' if plausible else 'ruled out (max angle exceeds it)'}"
              + (f" - {note}" if note else ""))

    if common_step:
        implied_spokes_per_rotation_guess = None
        for candidate in (1024, 2048, 4096, 8192):
            if candidate > max_angle and candidate % common_step == 0:
                implied_spokes_per_rotation_guess = candidate // common_step
        if implied_spokes_per_rotation_guess:
            print(f"\nIf ANGLE_DIVISIONS were the smallest plausible candidate above, "
                  f"spokes/rotation needed for full coverage = ANGLE_DIVISIONS / step.")

    # Most direct empirical answer: spokes per frame * frames needed to see
    # the angle counter wrap around back near its starting value.
    print(f"\n--- Direct empirical check (no assumptions) ---")
    print("Looking for wraparound (angle decreasing sharply) across the whole capture...")
    wraps = 0
    prev = None
    spoke_count_since_wrap = 0
    spoke_counts_per_wrap = []
    for a in all_angles:
        if prev is not None and a < prev - 1000:  # sharp drop = wrapped around 0
            wraps += 1
            spoke_counts_per_wrap.append(spoke_count_since_wrap)
            spoke_count_since_wrap = 0
        spoke_count_since_wrap += 1
        prev = a

    if wraps > 0:
        print(f"Observed {wraps} wraparound(s) in this capture.")
        print(f"Spoke counts between wraps: {spoke_counts_per_wrap}")
        print(f">>> This is the actual number of spokes per rotation for YOUR unit. <<<")
    else:
        print("No wraparound observed in this file (capture may be shorter than one "
              "full rotation, or too sparse/throttled to catch it).")
        print("Try this on a denser/longer capture if possible.")

    print(f"\nRaw angle values (first 40): {all_angles[:40]}")


if __name__ == "__main__":
    main()