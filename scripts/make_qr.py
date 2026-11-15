"""Generate and verify print-grade QR code assets for TrustFed-SDN (Step 10).

Implements full ISO/IEC 18004 QR Code encoding (Byte mode, GF(256) Reed-Solomon
ECC, optimal 8-mask penalty selection) and independent image-grid decoding with
BCH(15,5) format validation and Reed-Solomon zero-syndrome verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://trustfed.naimurrashid.dev/"
DEFAULT_OUT_DIR = REPO_ROOT / "site" / "assets" / "qr"
SITE_MIRROR_QR_DIR = REPO_ROOT.parent / "07_Website" / "trustfed-sdn-site" / "images" / "qr"

# ECC indicator 2-bit codes in ISO/IEC 18004 Format Information
ECC_FORMAT_BITS = {"L": 0b01, "M": 0b00, "Q": 0b11, "H": 0b10}
BITS_TO_ECC = {v: k for k, v in ECC_FORMAT_BITS.items()}

# Alignment center coordinate lists per QR Version (ISO/IEC 18004 Table E.1)
ALIGNMENT_POSITIONS: dict[int, list[int]] = {
    1: [],
    2: [6, 18],
    3: [6, 22],
    4: [6, 26],
    5: [6, 30],
    6: [6, 34],
}

# (total_data_cw, g1_blocks, g1_data_cw, g2_blocks, g2_data_cw, ec_cw_per_block)
EC_TABLE: dict[tuple[int, str], tuple[int, int, int, int, int, int]] = {
    (1, "L"): (19, 1, 19, 0, 0, 7),
    (1, "M"): (16, 1, 16, 0, 0, 10),
    (1, "Q"): (13, 1, 13, 0, 0, 13),
    (1, "H"): (9, 1, 9, 0, 0, 17),
    (2, "L"): (34, 1, 34, 0, 0, 10),
    (2, "M"): (28, 1, 28, 0, 0, 16),
    (2, "Q"): (22, 1, 22, 0, 0, 22),
    (2, "H"): (16, 1, 16, 0, 0, 28),
    (3, "L"): (55, 1, 55, 0, 0, 15),
    (3, "M"): (44, 1, 44, 0, 0, 26),
    (3, "Q"): (34, 2, 17, 0, 0, 18),
    (3, "H"): (26, 2, 13, 0, 0, 22),
    (4, "L"): (80, 1, 80, 0, 0, 20),
    (4, "M"): (64, 2, 32, 0, 0, 18),
    (4, "Q"): (48, 2, 24, 0, 0, 26),
    (4, "H"): (36, 4, 9, 0, 0, 16),
    (5, "L"): (108, 1, 108, 0, 0, 26),
    (5, "M"): (86, 2, 43, 0, 0, 24),
    (5, "Q"): (62, 2, 15, 2, 16, 18),
    (5, "H"): (46, 2, 11, 2, 12, 22),
    (6, "L"): (136, 2, 68, 0, 0, 18),
    (6, "M"): (108, 4, 27, 0, 0, 16),
    (6, "Q"): (76, 4, 19, 0, 0, 24),
    (6, "H"): (60, 4, 15, 0, 0, 28),
}


def _init_gf256() -> tuple[list[int], list[int]]:
    """Precompute GF(256) log/exp tables under primitive poly 0x11D."""
    gf_exp = [0] * 512
    gf_log = [0] * 256
    x = 1
    for i in range(255):
        gf_exp[i] = x
        gf_log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    for i in range(255, 512):
        gf_exp[i] = gf_exp[i - 255]
    return gf_exp, gf_log


_GF_EXP, _GF_LOG = _init_gf256()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _rs_generator_poly(degree: int) -> list[int]:
    """Compute Reed-Solomon generator polynomial coefficients (high to low)."""
    poly = [1]
    for i in range(degree):
        # Multiply poly by (x - alpha^i) = (x + alpha^i) in GF(256)
        factor = [1, _GF_EXP[i]]
        res = [0] * (len(poly) + 1)
        for p_idx, p_coef in enumerate(poly):
            for f_idx, f_coef in enumerate(factor):
                res[p_idx + f_idx] ^= _gf_mul(p_coef, f_coef)
        poly = res
    return poly


def _rs_encode_block(data: list[int], ec_count: int) -> list[int]:
    """Return `ec_count` Reed-Solomon error-correction bytes for `data`."""
    gen = _rs_generator_poly(ec_count)
    buf = list(data) + [0] * ec_count
    for i in range(len(data)):
        coef = buf[i]
        if coef != 0:
            for j in range(1, len(gen)):
                buf[i + j] ^= _gf_mul(gen[j], coef)
    return buf[len(data) :]


def _rs_syndrome(codeword: list[int], ec_count: int) -> list[int]:
    """Evaluate Reed-Solomon syndromes S_0..S_{ec_count-1}; all zero iff clean."""
    syndromes: list[int] = []
    for root_idx in range(ec_count):
        val = 0
        root = _GF_EXP[root_idx]
        for byte_val in codeword:
            val = _gf_mul(val, root) ^ byte_val
        syndromes.append(val)
    return syndromes


def _format_bch15_5(ecc_level: str, mask_id: int) -> int:
    """Compute 15-bit BCH(15,5) masked format word per ISO/IEC 18004 Section 7.9."""
    data = (ECC_FORMAT_BITS[ecc_level] << 3) | mask_id
    d = data << 10
    gen = 0b10100110111
    for bit in range(14, 9, -1):
        if (d >> bit) & 1:
            d ^= gen << (bit - 10)
    raw_fmt = ((data << 10) | d) ^ 0b101010000010010
    return raw_fmt


def _decode_format_bits(raw15: int) -> tuple[str, int, int] | None:
    """Find closest valid (ecc_level, mask_id, bit_errors) from 15-bit format word."""
    best: tuple[int, str, int] | None = None
    for ecc in ("L", "M", "Q", "H"):
        for m in range(8):
            cand = _format_bch15_5(ecc, m)
            dist = (raw15 ^ cand).bit_count()
            if dist <= 3 and (best is None or dist < best[0]):
                best = (dist, ecc, m)
    if best is None:
        return None
    return best[1], best[2], best[0]


def _mask_condition(mask_id: int, r: int, c: int) -> bool:
    """Return True iff module (r, c) is inverted by ISO 18004 mask `mask_id`."""
    if mask_id == 0:
        return (r + c) % 2 == 0
    if mask_id == 1:
        return r % 2 == 0
    if mask_id == 2:
        return c % 3 == 0
    if mask_id == 3:
        return (r + c) % 3 == 0
    if mask_id == 4:
        return ((r // 2) + (c // 3)) % 2 == 0
    if mask_id == 5:
        return ((r * c) % 2 + (r * c) % 3) == 0
    if mask_id == 6:
        return (((r * c) % 2 + (r * c) % 3) % 2) == 0
    if mask_id == 7:
        return (((r + c) % 2 + (r * c) % 3) % 2) == 0
    raise ValueError(f"Unsupported mask id: {mask_id}")


def _build_function_grid(version: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (modules, is_function) grids with fixed patterns pre-populated."""
    size = 17 + 4 * version
    modules = np.zeros((size, size), dtype=np.uint8)
    is_func = np.zeros((size, size), dtype=bool)

    def place_finder(top: int, left: int) -> None:
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = top + dr, left + dc
                if 0 <= r < size and 0 <= c < size:
                    is_func[r, c] = True
                    if 0 <= dr <= 6 and 0 <= dc <= 6:
                        on_border = dr in (0, 6) or dc in (0, 6)
                        in_core = 2 <= dr <= 4 and 2 <= dc <= 4
                        modules[r, c] = 1 if (on_border or in_core) else 0
                    else:
                        modules[r, c] = 0

    place_finder(0, 0)
    place_finder(0, size - 7)
    place_finder(size - 7, 0)

    # Alignment patterns
    centers = ALIGNMENT_POSITIONS[version]
    for r0 in centers:
        for c0 in centers:
            # Skip corners occupied by finder patterns
            if (
                (r0 <= 8 and c0 <= 8)
                or (r0 <= 8 and c0 >= size - 9)
                or (r0 >= size - 9 and c0 <= 8)
            ):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = r0 + dr, c0 + dc
                    is_func[r, c] = True
                    on_ring = max(abs(dr), abs(dc)) == 2
                    is_center = dr == 0 and dc == 0
                    modules[r, c] = 1 if (on_ring or is_center) else 0

    # Timing patterns (row 6 and column 6)
    for i in range(8, size - 8):
        val = 1 if i % 2 == 0 else 0
        if not is_func[6, i]:
            is_func[6, i] = True
            modules[6, i] = val
        if not is_func[i, 6]:
            is_func[i, 6] = True
            modules[i, 6] = val

    # Dark module at (4*version + 9, 8)
    is_func[4 * version + 9, 8] = True
    modules[4 * version + 9, 8] = 1

    # Reserve Format Information cells
    for i in range(9):
        if i != 6:
            is_func[8, i] = True
            is_func[i, 8] = True
    for i in range(8):
        is_func[8, size - 1 - i] = True
    for i in range(7):
        is_func[size - 1 - i, 8] = True

    return modules, is_func


def _write_format_info(modules: np.ndarray, ecc_level: str, mask_id: int) -> None:
    """Write both redundant copies of 15-bit BCH format word into `modules`."""
    size = modules.shape[0]
    fmt = _format_bch15_5(ecc_level, mask_id)

    # Copy 1: LSB (bit 0) -> (0, 8) down to MSB (bit 14) -> (8, 0)
    bits = [(fmt >> i) & 1 for i in range(15)]

    # Vertical top-left (bits 0..5 at rows 0..5, bit 6 at row 7, bit 7 at row 8)
    for idx, row in enumerate([0, 1, 2, 3, 4, 5, 7, 8]):
        modules[row, 8] = bits[idx]
    # Horizontal top-left (bit 8 at col 7, bits 9..14 at cols 5..0)
    for offset, col in enumerate([7, 5, 4, 3, 2, 1, 0]):
        modules[8, col] = bits[8 + offset]

    # Copy 2: bits 0..7 at (8, size-1..size-8), bits 8..14 at (size-7..size-1, 8)
    for i in range(8):
        modules[8, size - 1 - i] = bits[i]
    for i in range(7):
        modules[size - 7 + i, 8] = bits[8 + i]


def _zigzag_coords(is_func: np.ndarray) -> list[tuple[int, int]]:
    """Return data module coordinates in exact ISO 18004 upward/downward zigzag."""
    size = is_func.shape[0]
    coords: list[tuple[int, int]] = []
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for r in rows:
            for dc in (0, -1):
                c = col + dc
                if not is_func[r, c]:
                    coords.append((r, c))
        upward = not upward
        col -= 2
    return coords


def _penalty_score(mat: np.ndarray) -> int:
    """Compute ISO/IEC 18004 Section 7.8.3.1 mask penalty N1 + N2 + N3 + N4."""
    n = mat.shape[0]
    score = 0

    # N1: 5+ consecutive same-colored modules in rows and columns
    for line in list(mat) + list(mat.T):
        run = 1
        for i in range(1, n):
            if line[i] == line[i - 1]:
                run += 1
            else:
                if run >= 5:
                    score += 3 + (run - 5)
                run = 1
        if run >= 5:
            score += 3 + (run - 5)

    # N2: 2x2 uniform blocks
    for r in range(n - 1):
        for c in range(n - 1):
            s = int(mat[r, c]) + int(mat[r, c + 1]) + int(mat[r + 1, c]) + int(mat[r + 1, c + 1])
            if s in (0, 4):
                score += 3

    # N3: finder-like 1:1:3:1:1 patterns with 4 light modules on either side
    p1 = [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    p2 = [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    for line in list(mat) + list(mat.T):
        lst = line.tolist()
        for i in range(n - 10):
            seg = lst[i : i + 11]
            if seg == p1 or seg == p2:
                score += 40

    # N4: proportion of dark modules deviation from 50%
    dark = int(mat.sum())
    pct = (dark * 100.0) / (n * n)
    prev_mult = int(abs(int(pct // 5) * 5 - 50) / 5)
    next_mult = int(abs((int(pct // 5) * 5 + 5) - 50) / 5)
    score += min(prev_mult, next_mult) * 10

    return score


def _choose_version(payload_len: int, ecc_level: str) -> int:
    """Select the smallest QR Version (1..6) fitting `payload_len` bytes under `ecc_level`."""
    needed_bits = 4 + 8 + 8 * payload_len
    for version in range(1, 7):
        total_data_cw = EC_TABLE[(version, ecc_level)][0]
        if total_data_cw * 8 >= needed_bits:
            return version
    raise ValueError(
        f"URL too long ({payload_len} B) for supported versions 1..6 at ECC={ecc_level}"
    )


@dataclass
class QREncodeResult:
    url: str
    version: int
    ecc_level: str
    mask_id: int
    matrix: np.ndarray  # shape (N, N), values 0 or 1


def encode_qr(url: str, ecc_level: str = "H") -> QREncodeResult:
    """Encode `url` in ISO/IEC 18004 Byte mode with Reed-Solomon ECC."""
    raw_bytes = url.encode("utf-8")
    version = _choose_version(len(raw_bytes), ecc_level)
    total_data_cw, g1_blk, g1_len, g2_blk, g2_len, ec_len = EC_TABLE[(version, ecc_level)]

    # 1. Build bit stream: 0100 (Byte mode) + 8-bit char count + bytes + terminator + pads
    bits: list[int] = []

    def push_bits(val: int, width: int) -> None:
        for b in range(width - 1, -1, -1):
            bits.append((val >> b) & 1)

    push_bits(0b0100, 4)
    push_bits(len(raw_bytes), 8)
    for byte in raw_bytes:
        push_bits(byte, 8)

    max_bits = total_data_cw * 8
    term_len = min(4, max_bits - len(bits))
    bits.extend([0] * term_len)
    while len(bits) % 8 != 0:
        bits.append(0)

    data_cws: list[int] = []
    for i in range(0, len(bits), 8):
        val = 0
        for b in bits[i : i + 8]:
            val = (val << 1) | b
        data_cws.append(val)

    pad_cycle = [0xEC, 0x11]
    pad_idx = 0
    while len(data_cws) < total_data_cw:
        data_cws.append(pad_cycle[pad_idx % 2])
        pad_idx += 1

    # 2. Split into EC blocks & compute RS bytes
    blocks: list[list[int]] = []
    ec_blocks: list[list[int]] = []
    pos = 0
    for _ in range(g1_blk):
        blk = data_cws[pos : pos + g1_len]
        pos += g1_len
        blocks.append(blk)
        ec_blocks.append(_rs_encode_block(blk, ec_len))
    for _ in range(g2_blk):
        blk = data_cws[pos : pos + g2_len]
        pos += g2_len
        blocks.append(blk)
        ec_blocks.append(_rs_encode_block(blk, ec_len))

    # 3. Interleave data and EC codewords
    interleaved: list[int] = []
    max_data_len = max(len(b) for b in blocks)
    for i in range(max_data_len):
        for blk in blocks:
            if i < len(blk):
                interleaved.append(blk[i])
    for i in range(ec_len):
        for ec in ec_blocks:
            interleaved.append(ec[i])

    bitstream: list[int] = []
    for cw in interleaved:
        for b in range(7, -1, -1):
            bitstream.append((cw >> b) & 1)

    # 4. Place data bits and evaluate all 8 mask patterns
    base_mod, is_func = _build_function_grid(version)
    coords = _zigzag_coords(is_func)
    bit_idx = 0
    for r, c in coords:
        if bit_idx < len(bitstream):
            base_mod[r, c] = bitstream[bit_idx]
            bit_idx += 1
        else:
            base_mod[r, c] = 0

    best_mask = 0
    best_score = 10**9
    best_mat = base_mod.copy()

    for m in range(8):
        cand = base_mod.copy()
        for r, c in coords:
            if _mask_condition(m, r, c):
                cand[r, c] ^= 1
        _write_format_info(cand, ecc_level, m)
        sc = _penalty_score(cand)
        if sc < best_score:
            best_score = sc
            best_mask = m
            best_mat = cand

    return QREncodeResult(
        url=url,
        version=version,
        ecc_level=ecc_level,
        mask_id=best_mask,
        matrix=best_mat,
    )


def _extract_matrix_from_png(png_path: Path, expected_quiet: int | None = None) -> np.ndarray:
    """Sample a QR module grid (0/1) directly from a rendered QR PNG."""
    img = Image.open(png_path).convert("L")
    arr = np.asarray(img)
    dark_mask = arr < 128

    # Bounding box of dark modules (strips quiet zone automatically)
    ys, xs = np.where(dark_mask)
    if len(ys) == 0:
        raise ValueError(f"No dark modules detected in {png_path}")
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    cropped = dark_mask[r0 : r1 + 1, c0 : c1 + 1]

    # Find the horizontal 1:1:3:1:1 finder pattern width (= 7 modules) along rows
    # in the top-left finder region (robust to rounded outer corners on styled QRs).
    mod_px: float | None = None
    h_crop, w_crop = cropped.shape
    for scan_y in range(max(1, h_crop // 25), min(h_crop, h_crop // 5)):
        row_arr = cropped[scan_y, :]
        # Skip initial white pixels if rounded corner clipped (scan_y = 0)
        start_c = int(np.argmax(row_arr))
        if not row_arr[start_c]:
            continue
        runs: list[tuple[bool, int]] = []
        cur_val = bool(row_arr[start_c])
        cur_len = 0
        for px in row_arr[start_c:]:
            bpx = bool(px)
            if bpx == cur_val:
                cur_len += 1
            else:
                runs.append((cur_val, cur_len))
                if len(runs) == 5:
                    break
                cur_val = bpx
                cur_len = 1
        if len(runs) == 4:
            runs.append((cur_val, cur_len))
        if len(runs) == 5 and all(v for v, _ in runs[0::2]) and not any(v for v, _ in runs[1::2]):
            w_runs = [ln for _, ln in runs]
            unit = w_runs[0]
            # Check approx 1 : 1 : 3 : 1 : 1 ratio
            if unit > 0 and 2.2 * unit <= w_runs[2] <= 3.8 * unit and sum(w_runs) < w_crop * 0.45:
                mod_px = sum(w_runs) / 7.0
                break

    if mod_px is None or mod_px <= 0:
        raise ValueError(f"Could not detect 1:1:3:1:1 finder pattern in {png_path}")
    n_mods = round(cropped.shape[0] / mod_px)
    if n_mods < 21 or (n_mods - 17) % 4 != 0:
        raise ValueError(f"Estimated invalid QR dimension {n_mods} from {png_path}")

    cell_h = cropped.shape[0] / n_mods
    cell_w = cropped.shape[1] / n_mods
    mat = np.zeros((n_mods, n_mods), dtype=np.uint8)
    for r in range(n_mods):
        cy = min(cropped.shape[0] - 1, int((r + 0.5) * cell_h))
        for c in range(n_mods):
            cx = min(cropped.shape[1] - 1, int((c + 0.5) * cell_w))
            mat[r, c] = 1 if cropped[cy, cx] else 0
    return mat


def decode_and_verify_qr(png_path: Path, expected_url: str | None = None) -> dict[str, object]:
    """Decode a QR PNG grid, verify BCH format & Reed-Solomon zero syndromes, and check URL."""
    mat = _extract_matrix_from_png(png_path)
    n = mat.shape[0]
    version = (n - 17) // 4

    # Read Copy 1 and Copy 2 of Format Info
    fmt1 = 0
    c1_coords = [(0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (7, 8), (8, 8)] + [
        (8, 7),
        (8, 5),
        (8, 4),
        (8, 3),
        (8, 2),
        (8, 1),
        (8, 0),
    ]
    for bit_idx, (r, c) in enumerate(c1_coords):
        fmt1 |= int(mat[r, c]) << bit_idx

    fmt2 = 0
    c2_coords = [(8, n - 1 - i) for i in range(8)] + [(n - 7 + i, 8) for i in range(7)]
    for bit_idx, (r, c) in enumerate(c2_coords):
        fmt2 |= int(mat[r, c]) << bit_idx

    dec1 = _decode_format_bits(fmt1)
    dec2 = _decode_format_bits(fmt2)
    fmt_decoded = dec1 or dec2
    if fmt_decoded is None:
        raise ValueError(f"Failed BCH(15,5) format decoding for {png_path}")
    ecc_level, mask_id, _ = fmt_decoded

    _, is_func = _build_function_grid(version)
    coords = _zigzag_coords(is_func)
    raw_bits = [int(mat[r, c]) ^ (1 if _mask_condition(mask_id, r, c) else 0) for r, c in coords]

    total_data_cw, g1_blk, g1_len, g2_blk, g2_len, ec_len = EC_TABLE[(version, ecc_level)]
    total_cw = total_data_cw + (g1_blk + g2_blk) * ec_len

    cws: list[int] = []
    for i in range(0, total_cw * 8, 8):
        byte_val = 0
        for b in raw_bits[i : i + 8]:
            byte_val = (byte_val << 1) | b
        cws.append(byte_val)

    # De-interleave data and EC blocks
    blk_lens = [g1_len] * g1_blk + [g2_len] * g2_blk
    num_blocks = len(blk_lens)
    data_blocks: list[list[int]] = [[] for _ in range(num_blocks)]
    ec_blocks: list[list[int]] = [[] for _ in range(num_blocks)]

    idx = 0
    max_data_len = max(blk_lens)
    for col in range(max_data_len):
        for b_idx in range(num_blocks):
            if col < blk_lens[b_idx]:
                data_blocks[b_idx].append(cws[idx])
                idx += 1
    for _col in range(ec_len):
        for b_idx in range(num_blocks):
            ec_blocks[b_idx].append(cws[idx])
            idx += 1

    # Verify every Reed-Solomon block has zero syndrome
    for b_idx in range(num_blocks):
        syn = _rs_syndrome(data_blocks[b_idx] + ec_blocks[b_idx], ec_len)
        if any(syn):
            raise ValueError(f"Non-zero RS syndrome in block {b_idx} for {png_path}: {syn}")

    flat_data = [b for blk in data_blocks for b in blk]
    bit_str = "".join(f"{b:08b}" for b in flat_data)
    mode = int(bit_str[:4], 2)
    if mode != 0b0100:
        raise ValueError(f"Expected Byte mode (0100), got {bin(mode)}")
    char_count = int(bit_str[4:12], 2)
    payload_bytes = bytes(int(bit_str[12 + 8 * i : 20 + 8 * i], 2) for i in range(char_count))
    decoded_url = payload_bytes.decode("utf-8")

    if expected_url is not None and decoded_url != expected_url:
        raise AssertionError(f"Decoded QR URL mismatch: {decoded_url!r} != {expected_url!r}")

    return {
        "decoded_url": decoded_url,
        "version": version,
        "modules": n,
        "ecc_level": ecc_level,
        "mask_id": mask_id,
        "rs_blocks_verified": num_blocks,
        "rs_syndromes_zero": True,
    }


def _render_png(mat: np.ndarray, out_png: Path, quiet: int = 4, scale: int = 26) -> None:
    n = mat.shape[0]
    canvas_n = n + 2 * quiet
    grid = np.ones((canvas_n, canvas_n), dtype=np.uint8) * 255
    for r in range(n):
        for c in range(n):
            if mat[r, c]:
                grid[r + quiet, c + quiet] = 0
    img = Image.fromarray(grid, mode="L").resize(
        (canvas_n * scale, canvas_n * scale),
        resample=Image.Resampling.NEAREST,
    )
    img.convert("RGB").save(out_png, format="PNG")


def _render_svg(mat: np.ndarray, out_svg: Path, quiet: int = 4, cell: int = 10) -> None:
    n = mat.shape[0]
    total = (n + 2 * quiet) * cell
    rects: list[str] = []
    for r in range(n):
        for c in range(n):
            if mat[r, c]:
                x = (c + quiet) * cell
                y = (r + quiet) * cell
                rects.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}"/>')
    svg = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {total}" '
            f'width="{total}" height="{total}" shape-rendering="crispEdges">',
            f'  <rect width="{total}" height="{total}" fill="#FFFFFF"/>',
            '  <g fill="#0A0A0F">',
            *[f"    {line}" for line in rects],
            "  </g>",
            "</svg>",
        ]
    )
    out_svg.write_text(svg, encoding="utf-8")


def _render_pdf(mat: np.ndarray, out_pdf: Path, quiet: int = 4) -> None:
    n = mat.shape[0]
    total = n + 2 * quiet
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.set_xlim(0, total)
    ax.set_ylim(0, total)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), total, total, facecolor="#FFFFFF", edgecolor="none"))
    for r in range(n):
        for c in range(n):
            if mat[r, c]:
                ax.add_patch(
                    Rectangle(
                        (c + quiet, total - 1 - (r + quiet)),
                        1.0,
                        1.0,
                        facecolor="#0A0A0F",
                        edgecolor="none",
                    )
                )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def generate_qr_bundle(
    url: str = DEFAULT_URL,
    out_dir: Path = DEFAULT_OUT_DIR,
    ecc_level: str = "H",
) -> dict[str, object]:
    """Generate PNG/SVG/PDF QR assets, verify round-trip decoding, and write metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = encode_qr(url, ecc_level=ecc_level)

    out_png = out_dir / "trustfed_qr.png"
    out_svg = out_dir / "trustfed_qr.svg"
    out_pdf = out_dir / "trustfed_qr.pdf"

    _render_png(enc.matrix, out_png)
    _render_svg(enc.matrix, out_svg)
    _render_pdf(enc.matrix, out_pdf)

    decoded = decode_and_verify_qr(out_png, expected_url=url)
    summary = {
        "target_url": url,
        "version": enc.version,
        "modules": int(enc.matrix.shape[0]),
        "ecc_level": enc.ecc_level,
        "mask_id": enc.mask_id,
        "verification": decoded,
        "files": {
            "png": str(out_png.name),
            "svg": str(out_svg.name),
            "pdf": str(out_pdf.name),
        },
        "sha256": {
            "png": hashlib.sha256(out_png.read_bytes()).hexdigest(),
            "svg": hashlib.sha256(out_svg.read_bytes()).hexdigest(),
            "pdf": hashlib.sha256(out_pdf.read_bytes()).hexdigest(),
        },
    }
    (out_dir / "qr_verification.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and verify TrustFed-SDN QR code bundle.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Target URL for QR code.")
    parser.add_argument("--ecc", default="H", choices=["L", "M", "Q", "H"], help="ECC level.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output folder.")
    args = parser.parse_args(argv)

    summary = generate_qr_bundle(url=args.url, out_dir=args.out_dir, ecc_level=args.ecc)

    # Mirror into 07_Website/trustfed-sdn-site/images/qr/ if present
    if SITE_MIRROR_QR_DIR.parent.parent.exists():
        SITE_MIRROR_QR_DIR.mkdir(parents=True, exist_ok=True)
        for fname in [
            "trustfed_qr.png",
            "trustfed_qr.svg",
            "trustfed_qr.pdf",
            "qr_verification.json",
        ]:
            shutil.copy2(args.out_dir / fname, SITE_MIRROR_QR_DIR / fname)

    ver = summary["verification"]
    print(
        f"[Step 10] Generated QR v{summary['version']}-{summary['ecc_level']} "
        f"({summary['modules']}x{summary['modules']}, mask={summary['mask_id']}) -> "
        f"decoded & RS-verified: {ver['decoded_url']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
