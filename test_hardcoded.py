import struct
import math
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


# ── Constants ───────────────────────────────────────────────────────

SAMPLES_PER_SPOKE   = 1024   # 512 bytes * 2 nibbles each
BYTES_PER_SPOKE     = 512
NAVICO_HEADER_LEN   = 24     # 0x18 — common_header / br24_header / br4g_header
SPOKE_RECORD_LEN    = NAVICO_HEADER_LEN + BYTES_PER_SPOKE  # 536
FRAME_HDR_LEN       = 8      # radar_frame_pkt.frame_hdr
SPOKES_PER_FRAME    = 32
FRAME_LEN           = FRAME_HDR_LEN + SPOKES_PER_FRAME * SPOKE_RECORD_LEN  # 17160
ANGLE_DIVISIONS     = 4096   # full circle
MAX_INTENSITY       = 13     # nibble values 0-13 are intensity
DOPPLER_RECEDING    = 0xE
DOPPLER_APPROACHING = 0xF


# ── Header parsing (NavicoReceive.cpp common_header / br4g_header) ────────────────────────────────────────────────

def parse_spoke_header(data: bytes, offset: int = 0, radar_type: str = "halo"):
    """
    Parse a Navico spoke header at `offset`.
    Returns (header_len, angle_deg, range_meters) or None on failure.

    Layout (all types): byte 0 = headerLen (0x18), byte 1 = status (0x02/0x12).
    Angle: bytes 8-9 (LE). Halo range: br4g large @ 6-7, small @ 12-13.
    BR24 range: bytes 12-14 (3-byte LE), meters = raw * 10 / sqrt(2).
    """
    if len(data) < offset + NAVICO_HEADER_LEN + BYTES_PER_SPOKE:
        return None

    header_len = data[offset]
    status = data[offset + 1]

    if header_len != NAVICO_HEADER_LEN:
        return None
    if status not in (0x02, 0x12):
        return None

    angle_raw = struct.unpack_from("<H", data, offset + 8)[0]
    angle_deg = angle_raw * 360.0 / ANGLE_DIVISIONS

    if radar_type == "halo":
        large_range = struct.unpack_from("<H", data, offset + 6)[0]
        small_range = struct.unpack_from("<H", data, offset + 12)[0]
        if large_range == 0x80:
            range_m = 0 if small_range == 0xFFFF else small_range / 4.0
        else:
            range_m = large_range * small_range / 512.0
    else:
        r0, r1, r2 = data[offset + 12], data[offset + 13], data[offset + 14]
        range_raw = (r2 << 16) | (r1 << 8) | r0
        range_m = range_raw * 10.0 / math.sqrt(2) if range_raw > 0 else 0

    return header_len, angle_deg, range_m


def _spoke_valid(data: bytes, offset: int) -> bool:
    return parse_spoke_header(data, offset) is not None


# ── Nibble unpacking ───────────────────────────────────────────────────────

def unpack_nibbles(spoke_bytes: bytes) -> np.ndarray:
    """
    Unpack 512 bytes into 1024 4-bit samples.
    Low nibble = even bins, high nibble = odd bins.
    Returns uint8 array of length 1024 with values 0-15.
    """
    arr = np.frombuffer(spoke_bytes, dtype=np.uint8)
    out = np.empty(len(arr) * 2, dtype=np.uint8)
    out[0::2] = arr & 0x0F        # low nibble  → even index
    out[1::2] = (arr >> 4) & 0x0F # high nibble → odd index
    return out


# ── File readers ───────────────────────────────────────────────────────

def read_spokes_navico_frames(data: bytes) -> list:
    """
    Parse OpenCPN radar_raw.data blobs: frame_hdr[8] + up to 32 radar_line spokes.
    Multiple frames may be concatenated (export of many SQL rows).
    """
    spokes = []
    offset = 0
    n = len(data)

    while offset < n:
        spoke_start = offset
        if offset + FRAME_HDR_LEN + SPOKE_RECORD_LEN <= n and _spoke_valid(data, offset + FRAME_HDR_LEN):
            spoke_start = offset + FRAME_HDR_LEN
        elif not _spoke_valid(data, offset):
            offset += 1
            continue

        for _ in range(SPOKES_PER_FRAME):
            if spoke_start + SPOKE_RECORD_LEN > n:
                break
            if not _spoke_valid(data, spoke_start):
                break
            header_len, angle_deg, range_m = parse_spoke_header(data, spoke_start)
            data_start = spoke_start + header_len
            spoke_data = data[data_start:data_start + BYTES_PER_SPOKE]
            spokes.append((angle_deg, range_m, spoke_data))
            spoke_start += SPOKE_RECORD_LEN

        if spoke_start <= offset:
            offset += 1
        else:
            offset = spoke_start

    return spokes


def read_spokes_length_prefixed(data: bytes) -> list:
    """UDP payloads with optional 2-byte LE length before each packet."""
    spokes = []
    offset = 0
    n = len(data)

    while offset < n:
        if offset + 2 > n:
            break
        pkt_len = struct.unpack_from("<H", data, offset)[0]
        if not (8 < pkt_len < 65535) or offset + 2 + pkt_len > n:
            break
        offset += 2
        payload = data[offset:offset + pkt_len]
        offset += pkt_len
        spokes.extend(read_spokes_navico_frames(payload))

    return spokes


def detect_data_format(data: bytes) -> str:
    """Guess how bytes are framed."""
    n = len(data)
    if n == 0:
        return "empty"
    if n % FRAME_LEN == 0 and n >= FRAME_LEN and _spoke_valid(data, FRAME_HDR_LEN):
        return "frame"
    if n % SPOKE_RECORD_LEN == 0 and _spoke_valid(data, 0):
        return "spoke_stream"
    if n >= FRAME_HDR_LEN + SPOKE_RECORD_LEN and _spoke_valid(data, FRAME_HDR_LEN):
        return "frame"
    if n >= 2:
        first_len = struct.unpack_from("<H", data, 0)[0]
        if 8 < first_len < 65535 and first_len <= n - 2:
            trial = read_spokes_length_prefixed(data)
            if trial:
                return "length_prefixed"
    if n >= 512 * 512 * 3:
        return "img"
    return "unknown"


def read_spokes_raw(path: Path, fmt: str = "auto"):
    """
    Read Navico spokes from a binary file.

    fmt: auto | frame | raw | length_prefixed
    """
    data = path.read_bytes()
    if fmt == "auto":
        fmt = detect_data_format(data)

    if fmt == "frame" or fmt == "auto":
        spokes = read_spokes_navico_frames(data)
        if spokes:
            return spokes
    if fmt in ("raw", "length_prefixed", "auto"):
        spokes = read_spokes_length_prefixed(data)
        if spokes:
            return spokes
    if fmt == "auto":
        return read_spokes_navico_frames(data)
    return []


def read_radar_img_rgb(path: Path):
    """OpenCPN radar_img.rgb BLOB — typically 512x512x3 bytes (optional small header)."""
    data = path.read_bytes()
    for header_len in (0, 8, 16, 18, 24, 32):
        rem = len(data) - header_len
        if rem == 512 * 512 * 3:
            arr = np.frombuffer(data[header_len:], dtype=np.uint8).reshape(512, 512, 3)
            return arr
    return None


def read_spokes_pcap(path: Path):
    """Read spokes from a pcap file using scapy (optional dependency)."""
    try:
        from scapy.all import rdpcap, UDP
    except ImportError:
        sys.exit("scapy not installed. Run: pip install scapy --break-system-packages")

    pkts   = rdpcap(str(path))
    spokes = []
    for pkt in pkts:
        if UDP not in pkt:
            continue
        payload = bytes(pkt[UDP].payload)
        off = 0
        while off < len(payload):
            hdr = parse_spoke_header(payload, off)
            if hdr is None:
                break
            header_len, angle_deg, range_m = hdr
            data_start = off + header_len
            data_end   = data_start + BYTES_PER_SPOKE
            if data_end > len(payload):
                break
            spokes.append((angle_deg, range_m, payload[data_start:data_end]))
            off = data_end
    return spokes


# ── Polar → Cartesian rendering ────────────────────────────────────────────────

def render_ppi(spokes, image_size=1024, doppler=True):
    """
    Convert parsed spokes into a PPI radar image.

    Returns three numpy arrays (H, W):
        intensity  – float32, 0-1
        doppler_in – bool, approaching targets
        doppler_out – bool, receding targets
    """
    cx = cy = image_size // 2
    radius  = image_size // 2 - 4

    intensity   = np.zeros((image_size, image_size), dtype=np.float32)
    dop_app     = np.zeros((image_size, image_size), dtype=bool)
    dop_rec     = np.zeros((image_size, image_size), dtype=bool)

    # Build a meshgrid once for fast Cartesian→polar lookup
    xs = np.arange(image_size) - cx
    ys = np.arange(image_size) - cy
    XX, YY = np.meshgrid(xs, ys)

    for angle_deg, range_m, spoke_bytes in spokes:
        samples = unpack_nibbles(spoke_bytes)   # shape (1024,)
        n       = len(samples)

        angle_rad = math.radians(angle_deg)
        # Half-degree width per spoke for fill
        dtheta    = math.pi / ANGLE_DIVISIONS   # ≈ 0.044 deg in radians

        # For each spoke we scatter samples along the radial direction
        for i, val in enumerate(samples):
            if val == 0:
                continue
            r_frac = (i + 0.5) / n             # 0-1 fractional range
            px = cx + int(r_frac * radius * math.sin(angle_rad))
            py = cy - int(r_frac * radius * math.cos(angle_rad))

            if 0 <= px < image_size and 0 <= py < image_size:
                if val == DOPPLER_APPROACHING:
                    dop_app[py, px] = True
                elif val == DOPPLER_RECEDING:
                    dop_rec[py, px] = True
                else:
                    cur = intensity[py, px]
                    new = val / MAX_INTENSITY
                    intensity[py, px] = max(cur, new)

    return intensity, dop_app, dop_rec


def render_image(intensity, dop_app, dop_rec, range_m, image_size=1024):
    """
    Produce and save the final radar PPI image."""
    # Custom radar colour map: black → green → yellow → red
    colors_list = [
        (0.0, (0.0,  0.0,  0.0)),   # black  (no return)
        (0.2, (0.0,  0.5,  0.0)),   # dark green
        (0.5, (0.0,  1.0,  0.0)),   # green
        (0.75,(1.0,  1.0,  0.0)),   # yellow
        (1.0, (1.0,  0.0,  0.0)),   # red
    ]
    cmap = LinearSegmentedColormap.from_list(
        "radar", [(v, c) for v, c in colors_list]
    )

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")

    # Draw intensity
    ax.imshow(intensity, cmap=cmap, vmin=0, vmax=1,
              origin="upper", interpolation="nearest")

    # Overlay Doppler approaching (magenta) and receding (cyan)
    if dop_app.any():
        app_rgba = np.zeros((*dop_app.shape, 4), dtype=np.float32)
        app_rgba[dop_app] = [1.0, 0.0, 1.0, 1.0]  # magenta
        ax.imshow(app_rgba, origin="upper", interpolation="nearest")

    if dop_rec.any():
        rec_rgba = np.zeros((*dop_rec.shape, 4), dtype=np.float32)
        rec_rgba[dop_rec] = [0.0, 1.0, 1.0, 1.0]  # cyan
        ax.imshow(rec_rgba, origin="upper", interpolation="nearest")

    # Draw range rings
    cx = cy = image_size // 2
    r  = image_size // 2 - 4
    circle_kwargs = dict(fill=False, linestyle="--", linewidth=0.5, color="green", alpha=0.4)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        ring = plt.Circle((cx, cy), r * frac, **circle_kwargs)
        ax.add_patch(ring)
        if range_m > 0:
            ring_range_nm = (range_m * frac) / 1852
            ax.text(cx + r * frac + 2, cy, f"{ring_range_nm:.1f}nm",
                    color="green", fontsize=6, va="center", alpha=0.7)

    # Cardinal lines
    for angle_deg in range(0, 360, 30):
        rad = math.radians(angle_deg)
        ax.plot([cx, cx + r * math.sin(rad)],
                [cy, cy - r * math.cos(rad)],
                color="green", linewidth=0.3, alpha=0.3)
    for angle_deg, label in [(0,"N"),(90,"E"),(180,"S"),(270,"W")]:
        rad = math.radians(angle_deg)
        ax.text(cx + (r + 8) * math.sin(rad), cy - (r + 8) * math.cos(rad),
                label, color="white", fontsize=9, ha="center", va="center")

    ax.axis("off")
    plt.title(
        f"Navico Halo A  •  Range {range_m/1852:.2f} nm" if range_m > 0 else "Navico Halo A",
        color="white", fontsize=10, pad=4
    )
    plt.tight_layout(pad=0.5)
    #plt.savefig(output_path, dpi=150, facecolor="black")
    plt.show()
    #plt.close(fig)
    #print(f"Saved: {output_path}")


# ── Vectorised fast renderer (alternative, much faster for large files) ───────────────────────────

def render_ppi_fast(spokes, image_size=1024):
    """
    Faster polar→Cartesian renderer using numpy vectorisation.
    Returns (intensity, dop_app, dop_rec) arrays.
    """
    cx = cy = image_size // 2
    radius = image_size // 2 - 4

    intensity = np.zeros((image_size, image_size), dtype=np.float32)
    dop_app   = np.zeros((image_size, image_size), dtype=bool)
    dop_rec   = np.zeros((image_size, image_size), dtype=bool)

    bins   = np.arange(SAMPLES_PER_SPOKE, dtype=np.float32)
    r_frac = (bins + 0.5) / SAMPLES_PER_SPOKE

    for angle_deg, range_m, spoke_bytes in spokes:
        samples   = unpack_nibbles(spoke_bytes)
        angle_rad = math.radians(angle_deg)
        sin_a, cos_a = math.sin(angle_rad), math.cos(angle_rad)

        pxs = (cx + r_frac * radius * sin_a).astype(int)
        pys = (cy - r_frac * radius * cos_a).astype(int)

        mask = (pxs >= 0) & (pxs < image_size) & (pys >= 0) & (pys < image_size)

        valid_vals = samples[mask]
        valid_pxs  = pxs[mask]
        valid_pys  = pys[mask]

        app_mask = valid_vals == DOPPLER_APPROACHING
        rec_mask = valid_vals == DOPPLER_RECEDING
        int_mask = (valid_vals > 0) & ~app_mask & ~rec_mask

        dop_app[valid_pys[app_mask], valid_pxs[app_mask]] = True
        dop_rec[valid_pys[rec_mask], valid_pxs[rec_mask]] = True

        int_vals = (valid_vals[int_mask] / MAX_INTENSITY).astype(np.float32)
        np.maximum.at(intensity, (valid_pys[int_mask], valid_pxs[int_mask]), int_vals)

    return intensity, dop_app, dop_rec


# ── CLI ────────────────────────────────────────────────

def main():
    """
    parser = argparse.ArgumentParser(
        description="Decode Navico Halo A raw radar binary data and produce a PPI image."
    )
    parser.add_argument("--input", default="/content/drive/MyDrive/COllision/radardata.txt", help="Path to binary radar data file (default: /content/drive/MyDrive/COllision/radardata.txt)")
    parser.add_argument("--output", default="radar_ppi.png",
                        help="Output image path (default: radar_ppi.png)")
    parser.add_argument(
        "--format",
        choices=["auto", "frame", "raw", "pcap", "img"],
        default="auto",
        help="Input format (default: auto-detect)",
    )
    parser.add_argument("--size", type=int, default=1024,
                        help="Output image size in pixels (default: 1024)")
    parser.add_argument("--fast", action="store_true",
                        help="Use vectorised renderer (faster for large files)")
    args = parser.parse_args([]) # Modified this line

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    print(f"Reading {path} (format={args.format}) ...")
    if args.format == "pcap":
        spokes = read_spokes_pcap(path)
    elif args.format == "img":
        rgb = read_radar_img_rgb(path)
        if rgb is None:
            sys.exit(
                f"Not a 512x512 RGB radar_img blob ({path.stat().st_size} bytes). "
                "Use *raw.bin from radar_raw.data for spoke decoding."
            )
        out = Path(args.output)
        plt.imsave(str(out), rgb)
        print(f"Saved rendered OpenCPN image: {out}")
        return
    else:
        data = path.read_bytes()
        if args.format == "auto":
            detected = detect_data_format(data)
            print(f"Detected format: {detected}")
            if detected == "img":
                sys.exit(
                    f"This file looks like radar_img RGB ({len(data)} bytes), not radar_raw spokes. "
                    f"Run: python "Radar Image Generator.py" \"{path}\" --format img --output {args.output}"
                )
        spokes = read_spokes_raw(path, args.format if args.format != "auto" else "auto")

    if not spokes:
        sys.exit(
            "No spokes found.\n"
            "  *raw.bin  -> export radar_raw.data (UDP frame, ~17160 bytes per row)\n"
            "  *30.bin   -> likely radar_img.rgb (use --format img)\n"
            "  Or use --format pcap on a Wireshark capture."
        )

    # Use the most common range value across spokes
    ranges  = [r for _, r, _ in spokes if r > 0]
    range_m = float(np.median(ranges)) if ranges else 0.0

    print(f"Parsed {len(spokes)} spokes, range ~{range_m/1852:.2f} nm")
    """
    path = Path("sweep_1006.bin")
    print("Rendering PPI image ...")
    spokes = read_spokes_raw(path, "auto")
    ranges  = [r for _, r, _ in spokes if r > 0]
    range_m = float(np.median(ranges)) if ranges else 0.0
    intensity, dop_app, dop_rec = render_ppi(spokes, 1024)
    #render_image(intensity, dop_app, dop_rec, range_m, args.output, 1024)
    render_image(intensity, dop_app, dop_rec, range_m, 1024)

main()