"""
GC Trace Analyzer — Streamlit Web App
"""

import io
import zipfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

from gc_engine import (
    load_compound_library,
    load_gc_trace,
    analyze_sample,
    build_results_table,
    value_type_label,
    Compound,
    SampleResult,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="GC Multi-File Analyzer",
    page_icon="⚗️",
    layout="wide",
)

st.title("⚗️ GC Multi-File Analyzer")
st.caption("Upload GC traces, match compounds, and export publication-ready data.")

# ─────────────────────────────────────────────
# Sidebar — Compound Library
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("1 · Compound Library")
    lib_file = st.file_uploader(
        "Upload compound library — CSV (.csv) or Excel (.xlsx / .xls)",
        type=["csv", "xlsx", "xls"],
        key="lib",
    )
    st.markdown(
        """**Required columns:** `compound_name`, `retention_time`  
**Optional:** `functional_group`, `window` (±min, default 0.1), `mw` (g/mol),
`density` (g/mL), `is_internal_standard` (TRUE/FALSE), `response_factor` (default 1.0)  
📝 Excel: data must be on the **first sheet**."""
    )

    with st.expander("📄 Download template (CSV or Excel)"):
        template = pd.DataFrame({
            "compound_name": ["Dodecane (IS)", "Ethanol", "Butanol", "Acetone"],
            "functional_group": ["Alkane", "Alcohol", "Alcohol", "Ketone"],
            "retention_time": [5.00, 2.45, 3.82, 1.95],
            "window": [0.10, 0.10, 0.10, 0.10],
            "mw": [170.33, 46.07, 74.12, 58.08],
            "density": [0.749, 0.789, 0.810, 0.791],
            "is_internal_standard": [True, False, False, False],
            "response_factor": [1.0, 1.0, 1.0, 1.0],
        })
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.download_button(
                "⬇ Download as CSV",
                template.to_csv(index=False),
                "compound_library_template.csv",
                "text/csv",
                use_container_width=True,
            )
        with t_col2:
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                template.to_excel(writer, index=False, sheet_name="Compounds")
            st.download_button(
                "⬇ Download as Excel",
                xlsx_buf.getvalue(),
                "compound_library_template.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.divider()
    st.header("2 · Internal Standard")
    is_moles_input = st.number_input(
        "Moles of IS added (mol) — leave 0 to skip yield calculations",
        min_value=0.0,
        value=0.0,
        format="%.6f",
        step=1e-6,
    )
    is_moles = is_moles_input if is_moles_input > 0 else None

    st.divider()
    st.header("3 · Peak Detection Settings")
    with st.expander("Advanced peak parameters", expanded=False):
        min_height_pct = st.slider("Min peak height (% of max signal)", 0.1, 10.0, 0.5, 0.1)
        min_prom_pct = st.slider("Min prominence (% of max signal)", 0.1, 10.0, 0.2, 0.1)
        min_width_pts = st.slider("Min peak width (data points)", 2, 20, 3, 1)
        smooth_window = st.slider("Smoothing window (points, must be odd)", 3, 51, 11, 2)
        baseline_window_pct = st.slider("Baseline window (% of trace)", 1, 20, 5, 1)
        integration_rel_height = st.slider(
            "Integration boundary depth (% of peak height)",
            min_value=50, max_value=99, value=99, step=1,
            help=(
                "How far down each peak side to set the integration boundary. "
                "99% = nearly to the base (wider). 80% = stops higher up (tighter). "
                "Reduce this if boundaries are bleeding into neighboring peaks."
            ),
        )
        min_area_pct = st.slider(
            "Min peak area (% of largest peak area)",
            min_value=0.01, max_value=10.0, value=0.5, step=0.01,
            help=(
                "Peaks whose integrated area is below this fraction of the largest "
                "peak are discarded as noise. Increase this to suppress false peaks "
                "in flat baseline regions at the end of the trace."
            ),
        )

    peak_params = dict(
        min_height_pct=min_height_pct,
        min_prominence_pct=min_prom_pct,
        min_width_pts=min_width_pts,
        smooth_window=smooth_window,
        baseline_window_pct=baseline_window_pct / 100,
        integration_rel_height=integration_rel_height / 100,
        min_area_pct=min_area_pct,
    )

# ─────────────────────────────────────────────
# Main — GC Trace Upload
# ─────────────────────────────────────────────

st.header("4 · Upload GC Traces")
trace_files = st.file_uploader(
    "Upload one or more GC trace files — CSV (.csv) or Excel (.xlsx / .xls) — each file = one sample",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key="traces",
)

# ─────────────────────────────────────────────
# Output settings
# ─────────────────────────────────────────────

st.header("5 · Output Settings")
col1, col2 = st.columns(2)

with col1:
    output_modes = st.multiselect(
        "Group results by (select one or both — each gets its own file)",
        ["Compound", "Functional Group"],
        default=["Compound", "Functional Group"],
    )

with col2:
    value_type_labels = {
        "raw_area":          "Raw Peak Area",
        "normalized_area":   "Normalized Area (÷ IS ÷ RF)",
        "molar_yield":       "Molar Yield (mmol)",
        "mass_yield":        "Mass Yield (mg)",
        "fg_fraction":       "Functional Group Fraction (of total area)",
        "compound_fraction": "Compound Fraction (of total area)",
    }
    value_types_available = ["raw_area", "compound_fraction", "normalized_area"]
    if is_moles:
        value_types_available += ["molar_yield", "mass_yield"]
    # fg_fraction only makes sense for functional group mode
    if "Functional Group" in (output_modes or []):
        value_types_available_fg = value_types_available + ["fg_fraction"]
    else:
        value_types_available_fg = value_types_available

    selected_value_types = st.multiselect(
        "Value types to export (each becomes a separate sheet)",
        options=value_types_available_fg,
        format_func=lambda x: value_type_labels[x],
        default=value_types_available_fg,
    )

show_overlay = st.checkbox("Show overlay plot of all traces", value=True)

# ─────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────

run = st.button("▶ Run Analysis", type="primary", use_container_width=True)

if run:
    # Validate inputs
    errors = []
    if not lib_file:
        errors.append("Please upload a compound library.")
    if not trace_files:
        errors.append("Please upload at least one GC trace file.")
    if not output_modes:
        errors.append("Please select at least one grouping mode.")
    if not selected_value_types:
        errors.append("Please select at least one value type.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # Load library
    with st.spinner("Loading compound library…"):
        try:
            compounds = load_compound_library(pd.read_csv(lib_file) if lib_file.name.endswith(".csv") else pd.read_excel(lib_file))
            st.success(f"✅ Loaded {len(compounds)} compounds ({sum(c.is_internal_standard for c in compounds)} IS)")
        except Exception as exc:
            st.error(f"Error loading compound library: {exc}")
            st.stop()

    # Load and analyze traces
    results: list[SampleResult] = []
    trace_data: dict[str, tuple] = {}  # name -> (time, signal)

    progress = st.progress(0, text="Analyzing traces…")
    for i, tf in enumerate(trace_files):
        sample_name = Path(tf.name).stem
        progress.progress((i + 1) / len(trace_files), text=f"Analyzing {sample_name}…")
        try:
            raw_bytes = tf.read()
            if tf.name.endswith((".xlsx", ".xls")):
                # Try with auto-detected header first; fall back to no-header
                df_raw = pd.read_excel(io.BytesIO(raw_bytes))
                numeric_cols = [c for c in df_raw.columns
                                if pd.to_numeric(df_raw[c], errors="coerce").notna().mean() > 0.8]
                if len(numeric_cols) < 2:
                    df_raw = pd.read_excel(io.BytesIO(raw_bytes), header=None)
            else:
                try:
                    df_raw = pd.read_csv(io.BytesIO(raw_bytes))
                except Exception:
                    df_raw = pd.read_csv(io.BytesIO(raw_bytes), header=None)
            time_arr, signal_arr = load_gc_trace(df_raw, sample_name)
            trace_data[sample_name] = (time_arr, signal_arr)
            result = analyze_sample(
                time_arr, signal_arr, compounds,
                sample_name=sample_name,
                is_moles=is_moles,
                peak_params=peak_params,
            )
            results.append(result)
        except Exception as exc:
            st.warning(f"⚠ Could not process '{tf.name}': {exc}")
            st.code(traceback.format_exc())

    progress.empty()

    if not results:
        st.error("No traces were successfully analyzed.")
        st.stop()

    st.success(f"✅ Analyzed {len(results)} sample(s).")

    # ─────────────────────────────────────────
    # Overlay plot
    # ─────────────────────────────────────────

    if show_overlay and trace_data:
        st.subheader("📈 GC Trace Overlay")
        n = len(trace_data)
        colors = cm.tab10(np.linspace(0, 1, min(n, 10)))

        fig, ax = plt.subplots(figsize=(12, 4))
        for idx, (sname, (t, s)) in enumerate(trace_data.items()):
            color = colors[idx % len(colors)]
            # Normalize each trace to its own max for overlay clarity
            s_norm = s / (s.max() if s.max() > 0 else 1)
            ax.plot(t, s_norm, label=sname, color=color, linewidth=0.9, alpha=0.85)

        # Annotate compound retention times
        for cmp in compounds:
            ls = "--" if cmp.is_internal_standard else ":"
            ax.axvline(cmp.retention_time, color="gray", linestyle=ls, linewidth=0.6, alpha=0.6)
            ax.text(cmp.retention_time, 1.02, cmp.name,
                    rotation=90, fontsize=6, va="bottom", ha="center", color="gray",
                    transform=ax.get_xaxis_transform())

        ax.set_xlabel("Retention Time (min)")
        ax.set_ylabel("Normalized Signal")
        ax.set_title("GC Trace Overlay (normalized to individual maxima)")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(-0.05, 1.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ─────────────────────────────────────────
    # Individual trace plots with peak annotations
    # ─────────────────────────────────────────

    with st.expander("🔬 Individual trace views (with detected peaks)", expanded=False):

        # Controls for what to show
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        with diag_col1:
            show_integration = st.checkbox("Show integration bounds", value=True)
        with diag_col2:
            show_rt_windows = st.checkbox("Show RT matching windows", value=True)
        with diag_col3:
            show_baseline = st.checkbox("Show baseline-corrected signal", value=False)

        for result in results:
            t, s = trace_data.get(result.sample_name, (None, None))
            if t is None:
                continue

            fig, ax = plt.subplots(figsize=(13, 4))

            # Raw signal
            ax.plot(t, s, color="steelblue", linewidth=0.9, label="Signal", zorder=3)

            # Optional: baseline-corrected signal
            if show_baseline:
                from gc_engine import baseline_correct
                s_corr = baseline_correct(s)
                ax.plot(t, s_corr, color="darkorange", linewidth=0.7,
                        linestyle="--", label="Baseline corrected", zorder=2, alpha=0.8)

            # RT matching windows for every compound in library
            if show_rt_windows:
                for cmp in compounds:
                    win_color = "green" if cmp.is_internal_standard else "purple"
                    ax.axvspan(
                        cmp.retention_time - cmp.window,
                        cmp.retention_time + cmp.window,
                        alpha=0.08, color=win_color, zorder=1,
                    )
                    ax.axvline(
                        cmp.retention_time,
                        color=win_color, linewidth=0.8,
                        linestyle="--", alpha=0.5, zorder=1,
                    )
                    # Label at bottom of plot
                    ax.text(
                        cmp.retention_time, 0.01, cmp.name,
                        rotation=90, fontsize=5.5, va="bottom", ha="center",
                        color=win_color, alpha=0.8,
                        transform=ax.get_xaxis_transform(),
                    )

            # Integration bounds + peak markers for detected peaks
            for peak in result.peaks:
                cmp = peak.compound
                # Integration shading
                if show_integration:
                    ax.axvspan(
                        t[peak.left_idx], t[peak.right_idx],
                        alpha=0.2, color="orange", zorder=2,
                        label="_nolegend_",
                    )
                    # Integration boundary lines
                    ax.axvline(t[peak.left_idx], color="orange",
                               linewidth=0.8, linestyle=":", alpha=0.8, zorder=2)
                    ax.axvline(t[peak.right_idx], color="orange",
                               linewidth=0.8, linestyle=":", alpha=0.8, zorder=2)

                # Peak apex marker
                ax.plot(peak.retention_time, peak.height, "v",
                        color="red", markersize=6, zorder=4)

                # Compound label above apex
                name = cmp.name if cmp else "unmatched"
                rt_diff = f"\nΔRT={abs(peak.retention_time - cmp.retention_time):.3f}" if cmp else ""
                ax.annotate(
                    f"{name}{rt_diff}",
                    xy=(peak.retention_time, peak.height),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=6, ha="center", va="bottom",
                    rotation=90, color="darkred",
                )

            # Legend entries for windows
            if show_rt_windows:
                from matplotlib.patches import Patch
                legend_els = [
                    Patch(facecolor="purple", alpha=0.2, label="RT matching window"),
                    Patch(facecolor="green",  alpha=0.2, label="IS matching window"),
                ]
                if show_integration:
                    legend_els.append(Patch(facecolor="orange", alpha=0.3, label="Integration bounds"))
                ax.legend(handles=legend_els, fontsize=7, loc="upper right")

            ax.set_title(result.sample_name, fontsize=10, fontweight="bold")
            ax.set_xlabel("Retention Time (min)")
            ax.set_ylabel("Signal")
            ax.margins(x=0.02)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ─────────────────────────────────────────
    # Stacked integration view across all files
    # ─────────────────────────────────────────

    with st.expander("📊 Integration windows — all files stacked", expanded=False):
        st.caption(
            "Same x-axis across all samples. "
            "**Purple dashed lines** = RT matching window edges. "
            "**Orange shading** = actual integration bounds. "
            "**Red ▼** = detected peak apex."
        )

        diag2_col1, diag2_col2 = st.columns(2)
        with diag2_col1:
            stack_show_windows = st.checkbox("Show RT matching windows", value=True, key="stack_win")
        with diag2_col2:
            stack_show_integration = st.checkbox("Show integration bounds", value=True, key="stack_int")

        # Shared x-axis range across all traces
        all_times = [t for t, s in trace_data.values()]
        x_min = min(arr.min() for arr in all_times)
        x_max = max(arr.max() for arr in all_times)

        n_samples = len(results)
        fig, axes = plt.subplots(
            n_samples, 1,
            figsize=(13, 2.8 * n_samples),
            sharex=True,
        )
        if n_samples == 1:
            axes = [axes]

        colors = cm.tab10(np.linspace(0, 1, min(n_samples, 10)))

        for ax, result, color in zip(axes, results, colors):
            t, s = trace_data.get(result.sample_name, (None, None))
            if t is None:
                continue

            ax.plot(t, s, color=color, linewidth=0.9, zorder=3)

            # RT matching windows — just thin vertical lines at window edges, no fill
            if stack_show_windows:
                for cmp in compounds:
                    win_color = "green" if cmp.is_internal_standard else "purple"
                    # Window edge lines (dashed, thin)
                    ax.axvline(cmp.retention_time - cmp.window, color=win_color,
                               linewidth=0.6, linestyle="--", alpha=0.5, zorder=1)
                    ax.axvline(cmp.retention_time + cmp.window, color=win_color,
                               linewidth=0.6, linestyle="--", alpha=0.5, zorder=1)
                    # Center RT line (solid, thinner)
                    ax.axvline(cmp.retention_time, color=win_color,
                               linewidth=0.5, linestyle="-", alpha=0.25, zorder=1)

            # Integration bounds — tight orange shading only around actual peak
            for peak in result.peaks:
                if stack_show_integration:
                    ax.axvspan(t[peak.left_idx], t[peak.right_idx],
                               alpha=0.25, color="orange", zorder=2)
                    ax.axvline(t[peak.left_idx], color="darkorange",
                               linewidth=0.8, linestyle=":", alpha=0.9, zorder=2)
                    ax.axvline(t[peak.right_idx], color="darkorange",
                               linewidth=0.8, linestyle=":", alpha=0.9, zorder=2)
                ax.plot(peak.retention_time, peak.height, "v",
                        color="red", markersize=5, zorder=4)

            ax.set_ylabel(result.sample_name, fontsize=8, rotation=0,
                          ha="right", va="center", labelpad=60)
            ax.set_xlim(x_min, x_max)
            ax.tick_params(axis="y", labelsize=7)

        # Shared compound labels on top axis
        for cmp in compounds:
            axes[0].text(
                cmp.retention_time, 1.01, cmp.name,
                rotation=90, fontsize=5.5, va="bottom", ha="center",
                color="purple" if not cmp.is_internal_standard else "green",
                transform=axes[0].get_xaxis_transform(),
            )

        axes[-1].set_xlabel("Retention Time (min)", fontsize=9)
        fig.suptitle("Integration Windows — All Samples", fontsize=10, fontweight="bold", y=1.01)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ─────────────────────────────────────────
    # Peak exclusion editor
    # ─────────────────────────────────────────

    st.subheader("✏️ Review & Exclude Peaks")
    st.caption(
        "Uncheck any peak to exclude it from the results. "
        "For each excluded peak you can choose whether it appears as blank (NaN) or zero."
    )

    # Build a unique key for every detected peak: (sample_name, compound_name)
    # Store exclusion state and zero/nan choice in session_state
    if "excluded_peaks" not in st.session_state:
        st.session_state.excluded_peaks = {}   # key -> True if excluded
    if "excluded_as_zero" not in st.session_state:
        st.session_state.excluded_as_zero = {} # key -> True if zero, False if NaN

    # Build display rows
    peak_rows = []
    for result in results:
        for peak in result.peaks:
            cmp = peak.compound
            if cmp is None:
                continue
            peak_key = f"{result.sample_name}||{cmp.name}"
            peak_rows.append({
                "_key": peak_key,
                "Include": not st.session_state.excluded_peaks.get(peak_key, False),
                "Sample": result.sample_name,
                "Compound": cmp.name,
                "Functional Group": cmp.functional_group,
                "IS?": "✓" if cmp.is_internal_standard else "",
                "RT detected": round(peak.retention_time, 4),
                "RT library": round(cmp.retention_time, 4),
                "Δ RT": round(abs(peak.retention_time - cmp.retention_time), 4),
                "Area": round(peak.area, 4),
                "Height": round(peak.height, 4),
            })

    if peak_rows:
        edited = st.data_editor(
            pd.DataFrame(peak_rows).drop(columns=["_key"]),
            column_config={
                "Include": st.column_config.CheckboxColumn("Include", default=True),
                "IS?": st.column_config.TextColumn("IS?", disabled=True),
            },
            disabled=["Sample", "Compound", "Functional Group", "IS?",
                      "RT detected", "RT library", "Δ RT", "Area", "Height"],
            use_container_width=True,
            hide_index=True,
            key="peak_editor",
        )

        # Sync exclusions back to session state
        for i, row in edited.iterrows():
            peak_key = peak_rows[i]["_key"]
            st.session_state.excluded_peaks[peak_key] = not row["Include"]

        # For newly excluded peaks, ask zero or NaN
        newly_excluded = [
            peak_rows[i]["_key"]
            for i, row in edited.iterrows()
            if not row["Include"]
        ]
        if newly_excluded:
            st.markdown("**For each excluded peak — treat as:**")
            for pk in newly_excluded:
                sample, compound = pk.split("||")
                current = st.session_state.excluded_as_zero.get(pk, False)
                choice = st.radio(
                    f"`{sample}` → **{compound}**",
                    options=["Blank (NaN)", "Zero (0)"],
                    index=1 if current else 0,
                    horizontal=True,
                    key=f"zero_nan_{pk}",
                )
                st.session_state.excluded_as_zero[pk] = (choice == "Zero (0)")

    # Apply exclusions to a modified copy of results
    import copy
    results_filtered = copy.deepcopy(results)
    for result in results_filtered:
        for peak in result.peaks:
            if peak.compound is None:
                continue
            pk = f"{result.sample_name}||{peak.compound.name}"
            if st.session_state.excluded_peaks.get(pk, False):
                as_zero = st.session_state.excluded_as_zero.get(pk, False)
                if as_zero:
                    peak.area = 0.0
                    peak.height = 0.0
                else:
                    peak.compound = None   # NaN — treat as unmatched

        # Recalculate IS area after exclusions
        result.is_area = None
        for peak in result.peaks:
            if peak.compound and peak.compound.is_internal_standard:
                result.is_area = peak.area
                break

    # ─────────────────────────────────────────
    # Results tables + export (use filtered results)
    # ─────────────────────────────────────────

    st.subheader("📊 Results Tables")

    mode_map = {"Compound": "compound", "Functional Group": "functional_group"}
    mode_books: dict[str, dict[str, pd.DataFrame]] = {}

    for mode_label in output_modes:
        mode_key = mode_map[mode_label]
        mode_books[mode_label] = {}

        vt_list = [vt for vt in selected_value_types
                   if not (vt == "fg_fraction" and mode_key != "functional_group")
                   and not (vt == "compound_fraction" and mode_key != "compound")]

        st.markdown(f"---\n#### 📁 Grouped by {mode_label}")

        for vt in vt_list:
            try:
                df_out = build_results_table(results_filtered, output_mode=mode_key, value_type=vt)
                lbl = value_type_labels[vt]
                st.markdown(f"**{lbl}**")
                st.dataframe(
                    df_out.style.format(
                        subset=[c for c in df_out.columns if c != "File Name"],
                        formatter="{:.4f}",
                        na_rep="—",
                    ),
                    use_container_width=True,
                )
                sheet_name = lbl[:31]
                mode_books[mode_label][sheet_name] = df_out
            except Exception as exc:
                st.warning(f"Could not build {mode_label} / {vt} table: {exc}")
                st.code(traceback.format_exc())

    # ─────────────────────────────────────────
    # Export — one Excel workbook per mode + ZIP of all CSVs
    # ─────────────────────────────────────────

    st.subheader("💾 Export")

    # Build per-mode Excel files
    excel_files: dict[str, bytes] = {}   # filename -> bytes
    all_csv_files: dict[str, bytes] = {} # filename -> bytes

    for mode_label, sheets in mode_books.items():
        if not sheets:
            continue
        safe_mode = mode_label.lower().replace(" ", "_")
        fname = f"gc_results_{safe_mode}.xlsx"

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for sheet_name, df_sheet in sheets.items():
                df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        excel_files[fname] = buf.getvalue()

        for sheet_name, df_sheet in sheets.items():
            csv_name = f"{safe_mode}_{sheet_name[:40].replace(' ', '_')}.csv"
            all_csv_files[csv_name] = df_sheet.to_csv(index=False).encode()

    # Download buttons — one row per mode
    for fname, fbytes in excel_files.items():
        mode_label = "Compound" if "compound" in fname else "Functional Group"
        n_sheets = len(mode_books.get(mode_label, {}))
        st.download_button(
            f"⬇ {mode_label} results — Excel workbook ({n_sheets} sheet{'s' if n_sheets != 1 else ''})",
            fbytes,
            fname,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ZIP of all CSVs
    if all_csv_files:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for csv_name, csv_bytes in all_csv_files.items():
                zf.writestr(csv_name, csv_bytes)
        st.download_button(
            f"⬇ All results — ZIP of CSVs (Origin-ready, {len(all_csv_files)} files)",
            zip_buf.getvalue(),
            "gc_results_all.zip",
            "application/zip",
            use_container_width=True,
        )

    # Raw peak long-format CSV
    raw_rows = []
    for result in results:
        for peak in result.peaks:
            cmp = peak.compound
            raw_rows.append({
                "file_name": result.sample_name,
                "compound": cmp.name if cmp else "",
                "functional_group": cmp.functional_group if cmp else "",
                "is_internal_standard": cmp.is_internal_standard if cmp else False,
                "retention_time_detected": round(peak.retention_time, 4),
                "retention_time_library": cmp.retention_time if cmp else None,
                "area": round(peak.area, 6),
                "height": round(peak.height, 6),
                "is_area": round(result.is_area, 6) if result.is_area else None,
                "normalized_area": round(peak.area / result.is_area, 6)
                    if (result.is_area and cmp and not cmp.is_internal_standard) else None,
            })

    if raw_rows:
        raw_df = pd.DataFrame(raw_rows)
        st.download_button(
            "⬇ Raw peak data — long format CSV (all peaks, all samples)",
            raw_df.to_csv(index=False).encode(),
            "gc_raw_peaks.csv",
            "text/csv",
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.divider()
st.caption(
    "Peak detection: Savitzky-Golay smoothing → rolling-minimum baseline correction → "
    "scipy find_peaks (prominence + width filter) → trapezoidal integration. "
    "Compound matching: nearest retention time within user-defined window, "
    "best-area-first assignment (each compound assigned at most once per sample). "
    "Functional group fraction = group area / total non-IS area."
)

