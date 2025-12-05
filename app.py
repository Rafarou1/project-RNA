import streamlit as st
import os
import math
import tempfile
import shutil
import zipfile
import io
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# --- Shared Utilities ---
try:
    from rna_utils import (
        parse_pdb_atoms,
        get_bin_index,
        pair_key,
        load_params,
        PAIR_TYPES,
    )
except ImportError:
    st.error(
        "CRITICAL ERROR: 'rna_utils.py' not found. Please ensure it is in the same directory."
    )
    st.stop()

# --- Configuration ---
st.set_page_config(
    page_title="RNA Pipeline", layout="wide", initial_sidebar_state="expanded"
)


# --- NEW: Helper to Load Lottie ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# Scientific DNA/RNA Loading Animation
lottie_rna = load_lottieurl(
    "https://lottie.host/6e6403c9-0d6c-4aa7-933e-0014b2d499fa/R5Xv6p6XyK.json"
)

# --- CSS Styling (Updated for Dark Mode) ---
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Updated for Dark Theme Compatibility */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05); /* Transparent white for glass effect */
        border: 1px solid #414141;
        padding: 15px;
        border-radius: 8px;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "pipeline_run" not in st.session_state:
    st.session_state["pipeline_run"] = False
if "training_data_dir" not in st.session_state:
    st.session_state["training_data_dir"] = None
if "potentials_dir" not in st.session_state:
    st.session_state["potentials_dir"] = None
if "training_file_count" not in st.session_state:
    st.session_state["training_file_count"] = 0
if "score_history" not in st.session_state:
    st.session_state["score_history"] = []

# ==========================================
# CORE LOGIC ENGINES
# ==========================================


def run_training_engine(
    pdb_dir, out_dir, atom_type, max_dist, bin_width, status_container
):
    """Engine 1: Calculates statistics from PDBs."""
    nbins = int(math.ceil(max_dist / bin_width))
    pair_counts = {p: [0] * nbins for p in PAIR_TYPES}
    ref_counts = [0] * nbins

    files = []
    for root, dirs, filenames in os.walk(pdb_dir):
        for f in filenames:
            if f.lower().endswith(".pdb"):
                files.append(os.path.join(root, f))

    if not files:
        return False, "No PDB files found."

    total_files = len(files)
    processed_count = 0
    prog_bar = status_container.progress(0)

    for idx, fpath in enumerate(files):
        chains = parse_pdb_atoms(fpath, atom_type)
        for chain_id, residues in chains.items():
            n = len(residues)
            for i in range(n):
                for j in range(i + 4, n):
                    coords_i = residues[i][2]
                    coords_j = residues[j][2]
                    d = math.dist(coords_i, coords_j)
                    bin_idx = get_bin_index(d, max_dist, bin_width)
                    if bin_idx is not None:
                        key = pair_key(residues[i][1], residues[j][1])
                        if key in pair_counts:
                            pair_counts[key][bin_idx] += 1
                            ref_counts[bin_idx] += 1

        processed_count += 1
        prog_bar.progress((idx + 1) / total_files)

    EPS = 1e-12
    total_ref = sum(ref_counts) + EPS
    os.makedirs(out_dir, exist_ok=True)

    for pair in PAIR_TYPES:
        scores = []
        total_pair = sum(pair_counts[pair]) + EPS
        for i in range(nbins):
            p_obs = pair_counts[pair][i] / total_pair
            p_ref = ref_counts[i] / total_ref
            if p_obs < EPS:
                score = 10.0
            else:
                ratio = p_obs / (p_ref + EPS)
                score = -math.log(ratio) if ratio > 0 else 10.0
                score = min(max(score, -10.0), 10.0)
            scores.append(score)

        with open(os.path.join(out_dir, f"potential_{pair}.txt"), "w") as out_f:
            for s in scores:
                out_f.write(f"{s:.6f}\n")

        with open(os.path.join(out_dir, f"counts_{pair}.txt"), "w") as out_c:
            for c in pair_counts[pair]:
                out_c.write(f"{c}\n")

    with open(os.path.join(out_dir, "params.txt"), "w") as p_f:
        p_f.write(f"{atom_type}\n{max_dist}\n{bin_width}\n")

    return True, f"Processed {processed_count} files."


def run_plotting_engine(pot_dir, plot_type="Combined Overlay"):
    """Engine 2: Generates Plotly figures."""
    try:
        atom, max_dist, bin_width = load_params(pot_dir)
    except Exception as e:
        return None, f"Params missing: {e}"

    nbins = int(math.ceil(max_dist / bin_width))
    x_axis = [i * bin_width + (bin_width / 2) for i in range(nbins)]
    colors = px.colors.qualitative.Safe

    # --- Mode 1: Heatmap (NEW) ---
    if plot_type == "Heatmap Matrix":
        # Build 2D Matrix for Heatmap
        heatmap_data = []
        for pair in PAIR_TYPES:
            fname = os.path.join(pot_dir, f"potential_{pair}.txt")
            if os.path.exists(fname):
                with open(fname, "r") as f:
                    scores = [float(line.strip()) for line in f if line.strip()]
                # Pad or truncate if mismatch (safety)
                if len(scores) != len(x_axis):
                    scores = [0] * len(x_axis)
                heatmap_data.append(scores)
            else:
                heatmap_data.append([0] * len(x_axis))

        fig = px.imshow(
            heatmap_data,
            x=x_axis,
            y=PAIR_TYPES,
            labels=dict(x="Distance (Å)", y="Pair Type", color="Energy (kT)"),
            color_continuous_scale="RdBu_r",  # Red=High/Bad, Blue=Low/Good
            zmin=-4,
            zmax=4,  # Clamp for visual contrast
            aspect="auto",
        )
        fig.update_layout(
            title=f"Potential Landscape ({atom})", template="plotly_dark", height=600
        )
        return fig, None

    # --- Mode 2: Histogram ---
    elif plot_type == "Raw Counts (Histogram)":
        fig = make_subplots(
            rows=2,
            cols=5,
            subplot_titles=PAIR_TYPES,
            shared_xaxes=True,
            shared_yaxes=False,  # Y-axis not shared because counts vary wildly
            horizontal_spacing=0.03,
            vertical_spacing=0.1,
        )
        for idx, pair in enumerate(PAIR_TYPES):
            row = (idx // 5) + 1
            col = (idx % 5) + 1

            # Read the counts file we created in Step 1
            fname = os.path.join(pot_dir, f"counts_{pair}.txt")
            if os.path.exists(fname):
                with open(fname, "r") as f:
                    counts = [float(line.strip()) for line in f if line.strip()]

                if len(counts) == len(x_axis):
                    fig.add_trace(
                        go.Bar(
                            x=x_axis,
                            y=counts,
                            name=pair,
                            marker_color="#F59E0B",  # Orange color for contrast
                            showlegend=False,
                            hovertemplate=f"<b>{pair}</b><br>Dist: %{{x:.1f}}Å<br>Count: %{{y}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

        fig.update_layout(
            title=f"Raw Interaction Counts ({atom})", template="plotly_dark", height=700
        )
        return fig, None

    # --- Mode 3: Combined Overlay ---
    elif plot_type == "Combined Overlay":
        fig = go.Figure()
        for idx, pair in enumerate(PAIR_TYPES):
            fname = os.path.join(pot_dir, f"potential_{pair}.txt")
            if os.path.exists(fname):
                with open(fname, "r") as f:
                    scores = [float(line.strip()) for line in f if line.strip()]
                if len(scores) == len(x_axis):
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=scores,
                            mode="lines",
                            name=pair,
                            line=dict(width=2, color=colors[idx % len(colors)]),
                            hovertemplate=f"<b>{pair}</b><br>Dist: %{{x:.1f}}Å<br>Energy: %{{y:.2f}}<extra></extra>",
                        )
                    )
        fig.update_layout(
            title=f"Statistical Potentials ({atom}) - Overlay",
            xaxis_title="Distance (Å)",
            yaxis_title="Pseudo-energy (kT)",
            template="plotly_dark",  # UPDATED FOR DARK THEME
            hovermode="x unified",
            height=600,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_dist,
            y1=0,
            line=dict(color="white", width=1, dash="dash"),
        )
        fig.update_yaxes(range=[-11, 11])
        return fig, None

    # --- Mode 4: Grid View ---
    elif plot_type == "Grid View":
        fig = make_subplots(
            rows=2,
            cols=5,
            subplot_titles=PAIR_TYPES,
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.03,
            vertical_spacing=0.1,
        )
        for idx, pair in enumerate(PAIR_TYPES):
            row = (idx // 5) + 1
            col = (idx % 5) + 1
            fname = os.path.join(pot_dir, f"potential_{pair}.txt")
            if os.path.exists(fname):
                with open(fname, "r") as f:
                    scores = [float(line.strip()) for line in f if line.strip()]
                if len(scores) == len(x_axis):
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=scores,
                            mode="lines",
                            name=pair,
                            line=dict(width=2, color="#4ADE80"),  # Green for dark theme
                            showlegend=False,
                            hovertemplate=f"<b>{pair}</b><br>Dist: %{{x:.1f}}Å<br>Energy: %{{y:.2f}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=max_dist,
                y1=0,
                line=dict(color="white", width=1, dash="dash"),
                row=row,
                col=col,
            )
        fig.update_layout(
            title=f"Statistical Potentials ({atom}) - Grid View",
            template="plotly_dark",
            height=700,
        )
        fig.update_yaxes(range=[-11, 11])
        fig.update_xaxes(title_text="Distance (Å)", row=2, col=3)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        return fig, None


def run_scoring_engine(pdb_path, pot_dir):
    """Engine 3: Scores a structure."""
    try:
        atom_type, max_dist, bin_width = load_params(pot_dir)
    except Exception as e:
        return None, None, f"Params missing: {e}"

    potentials = {}
    for pair in PAIR_TYPES:
        fname = os.path.join(pot_dir, f"potential_{pair}.txt")
        if os.path.exists(fname):
            with open(fname, "r") as f:
                potentials[pair] = [float(line.strip()) for line in f if line.strip()]

    chains = parse_pdb_atoms(pdb_path, atom_type)
    if not chains:
        return 0.0, 0, f"No {atom_type} atoms found."

    total_score = 0.0
    pairs_used = 0

    def interp(dist, scores):
        if dist <= 0.0:
            return scores[0]
        if dist >= max_dist:
            return scores[-1]
        val_idx = dist / bin_width - 0.5
        idx_low = int(math.floor(val_idx))
        idx_high = idx_low + 1
        if idx_low <= 0:
            return scores[0]
        if idx_high >= len(scores):
            return scores[-1]
        frac = val_idx - idx_low
        return scores[idx_low] + (scores[idx_high] - scores[idx_low]) * frac

    for chain_id, residues in chains.items():
        n = len(residues)
        for i in range(n):
            for j in range(i + 4, n):
                d = math.dist(residues[i][2], residues[j][2])
                if d < max_dist:
                    key = pair_key(residues[i][1], residues[j][1])
                    if key in potentials:
                        total_score += interp(d, potentials[key])
                        pairs_used += 1

    return total_score, pairs_used, None


# ==========================================
# MAIN UI
# ==========================================


def main():
    # --- Sidebar ---
    with st.sidebar:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)
        else:
            st.markdown("## **RNA Potentials**")

        st.caption("v1.1 Alpha")

        # Navigation
        nav_selection = option_menu(
            menu_title=None,
            options=["Welcome", "Pipeline Dashboard"],
            icons=["house", "gear-wide-connected"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",
                },
                "icon": {"color": "#4ADE80", "font-size": "16px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#262730",
                },
                "nav-link-selected": {"background-color": "#262730"},
            },
        )

        st.divider()
        st.markdown("**Global Context**")

        # Initialize Session State
        if "atom_type" not in st.session_state:
            st.session_state["atom_type"] = "C3'"
        if "max_dist" not in st.session_state:
            st.session_state["max_dist"] = 20.0
        if "bin_width" not in st.session_state:
            st.session_state["bin_width"] = 1.0

        # Read-only context
        st.info(
            f"**Atom:** {st.session_state['atom_type']}\n\n"
            f"**Max:** {st.session_state['max_dist']} Å\n\n"
            f"**Bin:** {st.session_state['bin_width']} Å"
        )

    # --- PAGE: WELCOME ---
    if nav_selection == "Welcome":
        st.title("Welcome")

        # --- NEW: Lottie Animation Layout ---
        col_text, col_anim = st.columns([1.5, 1])

        with col_text:
            st.markdown("#### Statistical Potential Derivation Pipeline")
            st.markdown("""
            **1. Theory: Inverse Boltzmann Principle**
            The pipeline assumes that frequently observed structural features correspond to low-energy states. We calculate a pseudo-energy ($E$) for base pairs using the formula:
            
            $$
            E(r) = -kT \ln \\left( \\frac{P_{obs}(r)}{P_{ref}(r)} \\right)
            $$
            
            **2. Pipeline Logic**
            * **Training:** Extracts C3'-C3' distances (sequence separation $\ge$ 4).
            * **Visualisation:** Generates distance-dependent profiles for all 10 base-pair combinations.
            * **Scoring:** Scores new structures by summing interaction potentials.
            """)

        with col_anim:
            if lottie_rna:
                st_lottie(lottie_rna, height=300)

        st.info(
            """
            **Quick Start:**
            1. Drag PDB folder below.
            2. Go to 'Pipeline Dashboard'.
            3. Click 'Run'.
            """
        )

        st.divider()
        st.subheader("Data Deposit")

        uploaded_files = st.file_uploader(
            "Upload Training Dataset",
            accept_multiple_files=True,
            type=["pdb"],
            help="Drag a folder of PDB files here, or select multiple files.",
        )

        if uploaded_files:
            if not st.session_state["training_data_dir"]:
                st.session_state["training_data_dir"] = tempfile.mkdtemp(
                    prefix="rna_train_"
                )

            progress_text = st.empty()
            progress_text.text("Caching files...")

            error_count = 0

            for uploaded_file in uploaded_files:
                try:
                    file_path = os.path.join(
                        st.session_state["training_data_dir"], uploaded_file.name
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                except Exception:
                    error_count += 1

            saved_files = []
            for root, dirs, files in os.walk(st.session_state["training_data_dir"]):
                for file in files:
                    if file.lower().endswith(".pdb"):
                        saved_files.append(file)

            st.session_state["training_file_count"] = len(saved_files)

            progress_text.empty()
            if error_count > 0:
                st.warning(
                    f"Cached {len(saved_files)} files. (Skipped {error_count} problematic files)"
                )
            else:
                st.success(f"Successfully cached {len(saved_files)} PDB files.")

            st.markdown("Navigate to **Pipeline Dashboard** to proceed.")

        elif st.session_state["training_file_count"] > 0:
            st.info(
                f"{st.session_state['training_file_count']} files currently in memory."
            )

    # --- PAGE: PIPELINE DASHBOARD ---
    elif nav_selection == "Pipeline Dashboard":
        st.title("Pipeline Dashboard")

        # 1. CONFIGURATION
        with st.expander(
            "Configuration", expanded=not st.session_state["pipeline_run"]
        ):
            c1, c2, c3 = st.columns(3)
            # Keys bind these widgets directly to st.session_state
            c1.selectbox(
                "Atom Type", ["C3'", "P", "C4'", "C5'", "O3'"], key="atom_type"
            )
            c2.number_input("Max Distance (Å)", value=20.0, key="max_dist")
            c3.number_input("Bin Width (Å)", value=1.0, step=0.1, key="bin_width")

        # 2. EXECUTION
        col_exec, col_status = st.columns([1, 4])
        with col_exec:
            run_btn = st.button(
                "Run Pipeline", type="primary", use_container_width=True
            )

        with col_status:
            if st.session_state["training_file_count"] == 0:
                st.warning("No training data. Please deposit files in Welcome tab.")

        # RUN PIPELINE LOGIC
        if run_btn and st.session_state["training_file_count"] > 0:
            status_box = st.status("Processing...", expanded=True)

            status_box.write("Training Model...")
            tmp_out = tempfile.mkdtemp()

            success, msg = run_training_engine(
                st.session_state["training_data_dir"],
                tmp_out,
                st.session_state["atom_type"],
                st.session_state["max_dist"],
                st.session_state["bin_width"],
                status_box,
            )

            if success:
                persist_dir = os.path.join(
                    tempfile.gettempdir(), "rna_pipeline_results"
                )
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir)
                shutil.copytree(tmp_out, persist_dir)
                st.session_state["potentials_dir"] = persist_dir
                st.session_state["pipeline_run"] = True

                status_box.write("Finalizing...")
                status_box.update(
                    label="Pipeline Completed", state="complete", expanded=False
                )
                st.rerun()
            else:
                status_box.update(label="Pipeline Failed", state="error")
                st.error(msg)

        st.divider()

        # 3. RESULTS AREA
        if st.session_state["pipeline_run"] and st.session_state["potentials_dir"]:
            tab_viz, tab_score, tab_download = st.tabs(
                ["Visualisation", "Scoring", "Export"]
            )

            # --- TAB: VISUALISATION ---
            with tab_viz:
                c_sel, _ = st.columns([1, 3])
                with c_sel:
                    plot_choice = st.selectbox(
                        "View Mode",
                        [
                            "Combined Overlay",
                            "Grid View",
                            "Heatmap Matrix",
                            "Raw Counts (Histogram)",
                        ],
                    )

                fig, err = run_plotting_engine(
                    st.session_state["potentials_dir"], plot_type=plot_choice
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(err)

            # --- TAB: SCORING ---
            with tab_score:
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("##### Upload Structure")
                    target_pdb = st.file_uploader(
                        "Target PDB", type="pdb", label_visibility="collapsed"
                    )
                    score_trigger = st.button(
                        "Calculate Free Energy", disabled=not target_pdb
                    )

                with col_right:
                    st.markdown("##### Results")
                    if score_trigger and target_pdb:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdb"
                        ) as tmp:
                            tmp.write(target_pdb.getbuffer())
                            tpath = tmp.name

                        score, pairs, err = run_scoring_engine(
                            tpath, st.session_state["potentials_dir"]
                        )

                        if err:
                            st.error(err)
                        else:
                            # Color logic
                            if score < -10:
                                score_label = "Favourable"
                            elif score > 5:
                                score_label = "Unfavourable"
                            else:
                                score_label = "Neutral"

                            c1, c2 = st.columns(2)
                            c1.metric("Pseudo-Energy", f"{score:.4f}")
                            c2.metric("Interactions", pairs)

                            if score < 0:
                                st.success(score_label)
                            else:
                                st.warning(score_label)

                            # Append to History
                            st.session_state["score_history"].append(
                                {
                                    "Structure": target_pdb.name,
                                    "Pseudo-Energy": round(score, 4),
                                    "Interactions": pairs,
                                    "Classification": score_label,
                                }
                            )

                # History Table
                st.divider()
                st.markdown("##### Run History")

                if st.session_state["score_history"]:
                    df_history = pd.DataFrame(st.session_state["score_history"])
                    st.dataframe(df_history, use_container_width=True, hide_index=True)

                    csv_data = df_history.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download History (.csv)",
                        data=csv_data,
                        file_name="scoring_history.csv",
                        mime="text/csv",
                    )
                else:
                    st.caption("No scoring runs yet.")

            # --- TAB: DOWNLOADS ---
            with tab_download:
                st.write("Download the trained parameters and potential files.")
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zip_obj:
                    p_dir = st.session_state["potentials_dir"]
                    for f in os.listdir(p_dir):
                        zip_obj.write(os.path.join(p_dir, f), f)
                buf.seek(0)
                st.download_button(
                    label="Download Results (.zip)",
                    data=buf,
                    file_name="trained_potentials.zip",
                    mime="application/zip",
                )

        elif (
            not st.session_state["pipeline_run"]
            and st.session_state["training_file_count"] > 0
        ):
            st.info("Data loaded. Click 'Run Pipeline' above to generate results.")


if __name__ == "__main__":
    main()
