"""
Navico Halo A radar PPI from raw capture files.
Uses the decoder in "Radar Image Generator.py" (OpenCPN radar_pi / Navico spoke format).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from test import (
    BYTES_PER_SPOKE,
    detect_data_format,
    parse_spoke_header,
    read_spokes_raw,
    render_ppi_fast,
    render_image,
)

MAX_RANGE_M = 96_000  # ~52 nm, above typical Halo max scale


def filter_valid_spokes(spokes):
    """Drop false positives from misaligned or text-damaged bytes."""
    valid = []
    for angle_deg, range_m, spoke_bytes in spokes:
        if len(spoke_bytes) != BYTES_PER_SPOKE:
            continue
        if angle_deg < 0 or angle_deg >= 360.0:
            continue
        if range_m <= 0 or range_m > MAX_RANGE_M:
            continue
        valid.append((angle_deg, range_m, spoke_bytes))
    return valid


def diagnose_file(path: Path, scan_limit: int = 200_000) -> dict:
    """Check whether bytes look like intact Navico spokes or text-damaged."""
    data = path.read_bytes()
    n = len(data)
    space_frac = data.count(0x20) / n if n else 0.0
    null_frac = data.count(0x00) / n if n else 0.0

    # Scan for plausible spoke headers (header_len 8..64, angle 0..4096, sane range)
    header_hits = []
    step = 1 if n <= scan_limit else max(1, n // scan_limit)
    for off in range(0, min(n - BYTES_PER_SPOKE - 64, scan_limit), step):
        hdr = parse_spoke_header(data, off)
        if hdr is None:
            continue
        header_len, angle_deg, range_m = hdr
        if header_len not in (8, 24, 32, 48, 64):
            continue
        if not (0 <= angle_deg < 360.01):
            continue
        if range_m <= 0 or range_m > 100_000:
            continue
        spoke_end = off + header_len + BYTES_PER_SPOKE
        if spoke_end > n:
            continue
        header_hits.append((off, header_len, angle_deg, range_m))

    fmt = detect_data_format(path.read_bytes())
    spokes_raw = read_spokes_raw(path, "frame" if fmt == "frame" else "auto")
    spokes = filter_valid_spokes(spokes_raw)
    ranges = [r for _, r, _ in spokes if r > 0]
    angles = [a for a, _, _ in spokes]
    range_m = float(np.median(ranges)) if ranges else 0.0

    # Non-zero nibble fraction in first few spokes
    nonzero_frac = 0.0
    if spokes:
        from test import unpack_nibbles

        samples = []
        for _, _, sb in spokes[:20]:
            s = unpack_nibbles(sb)
            samples.append((s > 0).mean())
        nonzero_frac = float(np.mean(samples)) if samples else 0.0

    if fmt == "img":
        integrity = "radar_img_rgb_not_spokes"
    elif len(spokes) >= 8 and 0.001 < nonzero_frac < 0.95:
        integrity = "likely_intact"
    elif len(spokes_raw) > len(spokes):
        integrity = "false_positives_rejected"
    elif len(header_hits) > 10 and len(spokes) == 0:
        integrity = "headers_present_wrong_framing"
    elif space_frac > 0.85 or len(spokes) == 0:
        integrity = "text_damaged_or_wrong_format"
    else:
        integrity = "unknown_or_empty"

    return {
        "path": str(path),
        "size_bytes": n,
        "space_fraction": space_frac,
        "null_fraction": null_frac,
        "header_scan_hits": len(header_hits),
        "first_header_offsets": [h[0] for h in header_hits[:5]],
        "spokes_raw": len(spokes_raw),
        "spokes_parsed": len(spokes),
        "angle_min_deg": min(angles) if angles else None,
        "angle_max_deg": max(angles) if angles else None,
        "range_median_m": range_m,
        "range_nm": range_m / 1852.0 if range_m else 0.0,
        "nonzero_bin_fraction": nonzero_frac,
        "detected_format": fmt,
        "integrity": integrity,
    }


def print_report(report: dict) -> None:
    print(f"\n--- {Path(report['path']).name} ---")
    print(f"  Size:              {report['size_bytes']:,} bytes")
    print(f"  Space (0x20) frac: {report['space_fraction']:.1%}")
    print(f"  Null (0x00) frac:  {report['null_fraction']:.1%}")
    print(f"  Header scan hits:  {report['header_scan_hits']}")
    if report["first_header_offsets"]:
        print(f"  First offsets:     {[hex(o) for o in report['first_header_offsets']]}")
    print(f"  Spokes (raw/valid): {report['spokes_raw']} / {report['spokes_parsed']}")
    if report["spokes_parsed"]:
        print(f"  Angle range:       {report['angle_min_deg']:.2f} – {report['angle_max_deg']:.2f} deg")
        print(f"  Range (median):    {report['range_nm']:.2f} nm")
        print(f"  Non-zero bins:     {report['nonzero_bin_fraction']:.2%} (sample)")
    print(f"  Detected format:   {report['detected_format']}")
    print(f"  Verdict:           {report['integrity']}")


def main():
    parser = argparse.ArgumentParser(
        description="Decode Navico Halo raw .txt capture and render PPI (via Radar Image Generator.py)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="1241 847raw.txt",
        help="Raw capture file (default: 1241 847raw.txt)",
    )
    parser.add_argument(
        "--output",
        default="radar_from_generator1.png",
        help="Output PNG path",
    )
    parser.add_argument("--size", type=int, default=1024, help="PPI image size in pixels")
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Print integrity report only; do not render",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    report = diagnose_file(path)
    print_report(report)

    if args.diagnose_only:
        return

    if report["integrity"] == "radar_img_rgb_not_spokes":
        print("\nThis file is radar_img RGB. Decode with:")
        print(f'  python "Radar Image Generator.py" "{path}" --format img --output {args.output}')
        sys.exit(1)

    spokes = filter_valid_spokes(read_spokes_raw(path))
    if report["integrity"] != "likely_intact" or not spokes:
        print("\nCannot render: data failed integrity checks (see verdict above).")
        print("Use radar_raw.data exports (~17160 bytes/frame), not radar_img.rgb.")
        sys.exit(1)

    range_m = report["range_median_m"]
    print(f"\nRendering {len(spokes)} spokes -> {args.output}")
    intensity, dop_app, dop_rec = render_ppi_fast(spokes, args.size)
    render_image(intensity, dop_app, dop_rec, range_m, args.output, args.size)
    print("Done.")


if __name__ == "__main__":
    main()
