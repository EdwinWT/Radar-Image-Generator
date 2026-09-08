"""
Export OpenCPN radar_raw SQLite rows (time range) into one binary file
for "Radar Image Generator.py" / Radar Image Generator1.py.

Each row's data BLOB is one Navico UDP frame (8-byte hdr + 32 spokes).
Frames are written back-to-back in timestamp order.

Example:
    python export_radar_raw.py "../Navico Halo A/radarData.db" \\
        --start 2025-10-14T10:14:00 --end 2025-10-14T10:14:30 \\
        --output sweep_1014.bin

        python export_radar_raw.py radarData.db --start 2025-10-14T10:14:00 --end 2025-10-14T10:14:10 --output sweep_full.bin

    python "Radar Image Generator.py" sweep_1014.bin --output sweep_1014.png
"""

import argparse
import sqlite3
import sys
from pathlib import Path

FRAME_LEN = 8 + 32 * 536  # 17160 — Navico frame in radar_raw.data


def export_frames(db_path: Path, start: str, end: str, output: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, time, len, data
        FROM radar_raw
        WHERE time >= ? AND time <= ?
        ORDER BY time ASC, id ASC
        """,
        (start, end),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"rows": 0, "bytes": 0, "skipped": 0}

    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    skipped = 0
    lengths = set()

    with open(output, "wb") as out:
        for row_id, ts, declared_len, blob in rows:
            if blob is None:
                skipped += 1
                continue
            if declared_len and declared_len != len(blob):
                skipped += 1
                continue
            if len(blob) not in (FRAME_LEN,):
                # Still write non-standard blobs; decoder may skip invalid frames
                lengths.add(len(blob))
            out.write(blob)
            total += len(blob)

    return {
        "rows": len(rows),
        "frames_written": len(rows) - skipped,
        "skipped": skipped,
        "bytes": total,
        "non_standard_lengths": sorted(lengths),
        "output": str(output),
        "start": start,
        "end": end,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate radar_raw.data BLOBs for a time range into one .bin file."
    )
    parser.add_argument(
        "database",
        nargs="?",
        default="../Navico Halo A/radarData.db",
        help="Path to SQLite DB with radar_raw table",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start time (ISO), e.g. 2025-10-14T10:14:00",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End time (ISO), e.g. 2025-10-14T10:14:30",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="radar_sweep.bin",
        help="Output binary path (default: radar_sweep.bin)",
    )
    args = parser.parse_args()

    db_path = Path(args.database)
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path.resolve()}")

    output = Path(args.output)
    stats = export_frames(db_path, args.start, args.end, output)

    print(f"Database:  {db_path.resolve()}")
    print(f"Range:     {stats.get('start')} .. {stats.get('end')}")
    print(f"Rows:      {stats['rows']}")
    print(f"Written:   {stats['frames_written']} frames, {stats['bytes']:,} bytes")
    if stats["skipped"]:
        print(f"Skipped:   {stats['skipped']} (empty or len mismatch)")
    if stats.get("non_standard_lengths"):
        print(f"Warning:   non-{FRAME_LEN}-byte blobs: {stats['non_standard_lengths']}")
    if stats["bytes"] == 0:
        sys.exit("No data exported. Check time range and table contents.")

    spokes_est = (stats["frames_written"] * 32) if stats["frames_written"] else 0
    print(f"Output:    {output.resolve()}")
    print(f"~{spokes_est} spokes — run: python Radar Image Generator.py \"{output}\" --output sweep.png")


if __name__ == "__main__":
    main()
