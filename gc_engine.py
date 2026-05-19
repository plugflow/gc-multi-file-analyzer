"""
GC Trace Analysis Engine
Handles peak detection, integration, compound matching, and yield calculations.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class Compound:
    name: str
    functional_group: str
    retention_time: float      # minutes
    window: float              # ± minutes tolerance
    mw: Optional[float]        # g/mol
    density: Optional[float]   # g/mL (liquid, for mass yield)
    is_internal_standard: bool
    response_factor: float     # relative to IS (default 1.0)


@dataclass
class DetectedPeak:
    retention_time: float
    area: float
    height: float
    left_idx: int
    right_idx: int
    compound: Optional[Compound] = None


@dataclass
class SampleResult:
    sample_name: str
    peaks: list[DetectedPeak] = field(default_factory=list)
    is_area: Optional[float] = None
    is_moles: Optional[float] = None          # user-supplied moles of IS
    errors: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Compound library loader
# ─────────────────────────────────────────────

REQUIRED_COLS = {"compound_name", "retention_time"}

def load_compound_library(path_or_df) -> list[Compound]:
    """Load compound library from CSV/Excel path or a DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    elif str(path_or_df).endswith((".xlsx", ".xls")):
        df = pd.read_excel(path_or_df)
    else:
        df = pd.read_csv(path_or_df)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Compound library missing columns: {missing}")

    compounds = []
    for _, row in df.iterrows():
        compounds.append(Compound(
            name=str(row["compound_name"]).strip(),
            functional_group=str(row.get("functional_group", "Unknown")).strip(),
            retention_time=float(row["retention_time"]),
            window=float(row.get("window", 0.1)),
            mw=_safe_float(row.get("mw")),
            density=_safe_float(row.get("density")),
            is_internal_standard=_safe_bool(row.get("is_internal_standard", False)),
            response_factor=float(row.get("response_factor", 1.0)),
        ))
    return compounds


def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return v if not np.isnan(v) else None
    except (TypeError, ValueError):
        return None


def _safe_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in ("true", "yes", "1", "x")


# ─────────────────────────────────────────────
# GC trace loader
# ─────────────────────────────────────────────

def load_gc_trace(path_or_df, sample_name: str = "") -> tuple[np.ndarray, np.ndarray]:
    """
    Load a GC trace CSV/Excel and return (time_array, signal_array).
    Auto-detects time/signal columns.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    elif str(path_or_df).endswith((".xlsx", ".xls")):
        df = pd.read_excel(path_or_df, header=None)
    else:
        # Try with header first; fall back to no-header
        try:
            df = pd.read_csv(path_or_df)
            if df.shape[1] < 2:
                raise ValueError
        except Exception:
            df = pd.read_csv(path_or_df, header=None)

    df = df.dropna(how="all")

    # Auto-detect numeric columns
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    if len(numeric_cols) < 2:
        raise ValueError(f"Could not find two numeric columns in trace file for '{sample_name}'")

    time_col, signal_col = numeric_cols[0], numeric_cols[1]
    time = pd.to_numeric(df[time_col], errors="coerce").values.astype(float)
    signal = pd.to_numeric(df[signal_col], errors="coerce").values.astype(float)

    # Drop NaNs
    mask = np.isfinite(time) & np.isfinite(signal)
    time, signal = time[mask], signal[mask]

    # Sort by time
    order = np.argsort(time)
    return time[order], signal[order]


# ─────────────────────────────────────────────
# Peak detection & integration
# ─────────────────────────────────────────────

def baseline_correct(signal: np.ndarray, window_pct: float = 0.05) -> np.ndarray:
    """Rolling minimum baseline subtraction."""
    n = len(signal)
    w = max(int(n * window_pct), 5)
    # Pad-reflect to avoid edge effects
    padded = np.pad(signal, w, mode="reflect")
    baseline = np.array([padded[max(0, i):i + 2 * w].min() for i in range(n)])
    return signal - baseline


def _smooth(signal: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing."""
    w = min(window, len(signal) - 1)
    if w % 2 == 0:
        w -= 1
    if w < poly + 1:
        return signal.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return savgol_filter(signal, w, poly)


def detect_peaks(
    time: np.ndarray,
    signal: np.ndarray,
    min_height_pct: float = 0.5,        # % of max signal height
    min_prominence_pct: float = 0.2,    # % of max signal prominence
    min_width_pts: int = 3,
    smooth_window: int = 11,
    baseline_window_pct: float = 0.05,
    integration_rel_height: float = 0.99,
    min_area_pct: float = 0.01,         # % of largest peak area — discard below this
) -> list[DetectedPeak]:
    """
    Full pipeline:
      1. Baseline correct
      2. Smooth
      3. find_peaks with height/prominence/width filters
      4. Integrate each peak by trapezoidal rule within valley bounds
      5. Discard peaks whose area is below min_area_pct of the largest peak area
    """
    corrected = baseline_correct(signal, baseline_window_pct)
    smoothed = _smooth(corrected, smooth_window)

    sig_max = smoothed.max()
    if sig_max <= 0:
        return []

    min_h    = sig_max * (min_height_pct / 100)
    min_prom = sig_max * (min_prominence_pct / 100)

    peak_indices, properties = find_peaks(
        smoothed,
        height=min_h,
        prominence=min_prom,
        width=min_width_pts,
    )

    if len(peak_indices) == 0:
        return []

    # Get valley bounds using rel_height
    from scipy.signal import peak_widths
    rel_h = max(0.01, min(0.9999, integration_rel_height))
    widths, _, left_ips, right_ips = peak_widths(smoothed, peak_indices, rel_height=rel_h)

    detected = []
    for i, idx in enumerate(peak_indices):
        left_idx  = max(0, int(np.floor(left_ips[i])))
        right_idx = min(len(time) - 1, int(np.ceil(right_ips[i])))

        # Hard cap: never let a peak's bounds cross the midpoint to a neighboring peak
        if i > 0:
            prev_apex = peak_indices[i - 1]
            midpoint = (prev_apex + idx) // 2
            left_idx = max(left_idx, midpoint)
        if i < len(peak_indices) - 1:
            next_apex = peak_indices[i + 1]
            midpoint = (idx + next_apex) // 2
            right_idx = min(right_idx, midpoint)

        # Integrate on the corrected (not smoothed) signal
        t_seg = time[left_idx:right_idx + 1]
        s_seg = corrected[left_idx:right_idx + 1]
        area = float(trapezoid(s_seg, t_seg))

        detected.append(DetectedPeak(
            retention_time=float(time[idx]),
            area=max(area, 0.0),
            height=float(smoothed[idx]),
            left_idx=left_idx,
            right_idx=right_idx,
        ))

    # Post-integration area filter — drop noise peaks relative to largest real peak
    if detected:
        max_area = max(p.area for p in detected)
        min_area = max_area * (min_area_pct / 100)
        detected = [p for p in detected if p.area >= min_area]

    return detected


# ─────────────────────────────────────────────
# Compound matching
# ─────────────────────────────────────────────

def match_peaks_to_compounds(
    peaks: list[DetectedPeak],
    compounds: list[Compound],
) -> list[DetectedPeak]:
    """
    For each peak, find the closest compound within its retention window.
    Un-matched peaks are silently ignored (compound stays None).
    Each compound can only be assigned to one peak (best match wins).
    """
    used_compounds: set[str] = set()

    for peak in sorted(peaks, key=lambda p: p.area, reverse=True):
        best: Optional[Compound] = None
        best_dist = float("inf")
        for cmp in compounds:
            dist = abs(peak.retention_time - cmp.retention_time)
            if dist <= cmp.window and dist < best_dist and cmp.name not in used_compounds:
                best = cmp
                best_dist = dist
        if best:
            peak.compound = best
            used_compounds.add(best.name)

    return peaks


# ─────────────────────────────────────────────
# Analysis pipeline for a single sample
# ─────────────────────────────────────────────

def analyze_sample(
    time: np.ndarray,
    signal: np.ndarray,
    compounds: list[Compound],
    sample_name: str = "sample",
    is_moles: Optional[float] = None,
    peak_params: Optional[dict] = None,
) -> SampleResult:
    params = peak_params or {}
    result = SampleResult(sample_name=sample_name, is_moles=is_moles)

    peaks = detect_peaks(time, signal, **params)
    peaks = match_peaks_to_compounds(peaks, compounds)
    result.peaks = [p for p in peaks if p.compound is not None]

    # Find IS area
    for peak in result.peaks:
        if peak.compound and peak.compound.is_internal_standard:
            result.is_area = peak.area
            break

    return result


# ─────────────────────────────────────────────
# Results aggregation & yield calculations
# ─────────────────────────────────────────────

def build_results_table(
    results: list[SampleResult],
    output_mode: str = "compound",   # "compound" or "functional_group"
    value_type: str = "raw_area",    # "raw_area" | "normalized_area" | "molar_yield" | "mass_yield" | "fg_fraction"
) -> pd.DataFrame:
    """
    Build a wide-format DataFrame:
      rows    = one per sample
      columns = File Name | compound1 | compound2 | ...  (or functional groups)

    Functional group mode with value_type="fg_fraction":
      each cell = summed area of that group / total non-IS area  (fraction 0–1)
    """
    # Collect all column labels in library order (preserves RT order)
    seen: dict[str, None] = {}
    for r in results:
        for p in r.peaks:
            if p.compound and not p.compound.is_internal_standard:
                lbl = (p.compound.functional_group
                       if output_mode == "functional_group"
                       else p.compound.name)
                seen[lbl] = None
    all_labels = list(seen.keys())   # insertion-ordered = RT order

    rows = []
    for result in results:
        # Aggregate area per label for this sample
        label_area: dict[str, float] = {}
        label_compound: dict[str, Compound] = {}
        total_non_is_area = 0.0

        for peak in result.peaks:
            if peak.compound is None or peak.compound.is_internal_standard:
                continue
            lbl = (peak.compound.functional_group
                   if output_mode == "functional_group"
                   else peak.compound.name)
            label_area[lbl] = label_area.get(lbl, 0.0) + peak.area
            label_compound[lbl] = peak.compound
            total_non_is_area += peak.area

        row: dict = {"File Name": result.sample_name}
        for lbl in all_labels:
            raw = label_area.get(lbl, 0.0)
            cmp = label_compound.get(lbl)

            if value_type == "fg_fraction":
                # fraction of total detected (non-IS) area
                val = (raw / total_non_is_area) if (total_non_is_area > 0 and raw > 0) else (0.0 if raw == 0.0 else None)
            elif value_type == "compound_fraction":
                val = (raw / total_non_is_area) if (total_non_is_area > 0 and raw > 0) else (0.0 if raw == 0.0 else None)
            else:
                val = _compute_value(raw, cmp, result, value_type, lbl, output_mode)
            row[lbl] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("File Name")
    df.index.name = "File Name"
    # Reset so File Name is a real column (easier for Origin/Excel copy-paste)
    df = df.reset_index()
    return df


def _compute_value(
    raw_area: float,
    cmp: Optional[Compound],
    result: SampleResult,
    value_type: str,
    label: str,
    output_mode: str,
) -> Optional[float]:
    if value_type == "raw_area":
        return raw_area if raw_area > 0 else (0.0 if raw_area == 0.0 else None)

    if value_type == "normalized_area":
        if result.is_area and result.is_area > 0 and cmp:
            rf = cmp.response_factor if output_mode == "compound" else 1.0
            if raw_area > 0:
                return (raw_area / result.is_area) / rf
            elif raw_area == 0.0:
                return 0.0
        return None

    if value_type == "molar_yield":
        if not (result.is_area and result.is_area > 0 and result.is_moles and cmp and cmp.mw):
            return None
        rf = cmp.response_factor if output_mode == "compound" else 1.0
        if raw_area > 0:
            normalized = (raw_area / result.is_area) / rf
            return normalized * result.is_moles * 1000
        elif raw_area == 0.0:
            return 0.0
        return None

    if value_type == "mass_yield":
        if not (result.is_area and result.is_area > 0 and result.is_moles and cmp and cmp.mw):
            return None
        rf = cmp.response_factor if output_mode == "compound" else 1.0
        if raw_area > 0:
            normalized = (raw_area / result.is_area) / rf
            return normalized * result.is_moles * cmp.mw * 1000
        elif raw_area == 0.0:
            return 0.0
        return None

    return None


def value_type_label(value_type: str) -> str:
    return {
        "raw_area":          "Raw Peak Area (a.u.·min)",
        "normalized_area":   "Normalized Area (area / IS area / RF)",
        "molar_yield":       "Molar Yield (mmol)",
        "mass_yield":        "Mass Yield (mg)",
        "fg_fraction":       "Functional Group Fraction (fraction of total area)",
        "compound_fraction": "Compound Fraction (fraction of total area)",
    }.get(value_type, value_type)
