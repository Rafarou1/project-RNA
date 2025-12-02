import streamlit as st
import os
import math
import tempfile
import shutil
import zipfile
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# --- CSS Styling ---
st.markdown(
    """
    <style>
    /* Navigation Buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background-color: #f0f2f6;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        transition: all 0.3s;
        font-weight: 500;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #e6f3ff;
        border-color: #2e86de;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
        background-color: #2e86de;
        color: white;
        border-color: #2e86de;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radio"] > div:first-child {
        display: none;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #f9f9f9;
        border: 1px solid #eeeeee;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State ---
if "pipeline_run" not in st.session_state:
    st.session_state["pipeline_run"] = False
if "training_data_dir" not in st.session_state:
    st.session_state["training_data_dir"] = None
if "potentials_dir" not in st.session_state:
    st.session_state["potentials_dir"] = None
if "training_file_count" not in st.session_state:
    st.session_state["training_file_count"] = 0

# --- Core Logic (The Pipeline Engines) ---


def run_training_engine(
    pdb_dir, out_dir, atom_type, max_dist, bin_width, status_container
):
    """Engine 1: Calculates statistics from PDBs."""
    nbins = int(math.ceil(max_dist / bin_width))
    pair_counts = {p: [0] * nbins for p in PAIR_TYPES}
    ref_counts = [0] * nbins

    files = [
        os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb")
    ]
    if not files:
        return False, "No PDB files found."

    total_files = len(files)
    processed_count = 0

    # Progress Bar within the status container
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

    # Calculation
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

    # Save Params
    with open(os.path.join(out_dir, "params.txt"), "w") as p_f:
        p_f.write(f"{atom_type}\n{max_dist}\n{bin_width}\n")

    return True, f"Processed {processed_count} files."


def run_plotting_engine(pot_dir, plot_type="Combined Overlay"):
    """Engine 2: Generates Plotly figures (Combined or Grid)."""
    try:
        atom, max_dist, bin_width = load_params(pot_dir)
    except Exception as e:
        return None, f"Params missing : {e}"

    nbins = int(math.ceil(max_dist / bin_width))
    x_axis = [i * bin_width + (bin_width / 2) for i in range(nbins)]
    colors = px.colors.qualitative.Safe

    # --- Mode 1: Combined Overlay ---
    if plot_type == "Combined Overlay":
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
            template="simple_white",
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
            line=dict(color="black", width=1, dash="dash"),
        )
        fig.update_yaxes(range=[-11, 11])
        return fig, None

    # --- Mode 2: Grid View ---
    elif plot_type == "Grid View":
        # Create 2 rows x 5 columns
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
                            line=dict(width=2, color="blue"),  # Uniform color for grid
                            showlegend=False,
                            hovertemplate=f"<b>{pair}</b><br>Dist: %{{x:.1f}}Å<br>Energy: %{{y:.2f}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

            # Add zero line for each subplot
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=max_dist,
                y1=0,
                line=dict(color="black", width=1, dash="dash"),
                row=row,
                col=col,
            )

        fig.update_layout(
            title=f"Statistical Potentials ({atom}) - Grid View",
            template="simple_white",
            height=700,
        )
        fig.update_yaxes(range=[-11, 11])
        # Add axis labels to bottom left (approximate for shared axes)
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

    # Helper interpolation
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


# --- Main Application ---


def main():
    # --- Sidebar ---
    with st.sidebar:
        # Logo Check
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)
        else:
            st.markdown("## **RNA Potentials**")

        st.write("v.1.0 Alpha-Release")
        st.divider()

        # Navigation
        nav_selection = st.radio(
            "Navigation",
            ["Welcome Page", "Pipeline Dashboard"],
            label_visibility="collapsed",
        )

        # Global Params
        st.divider()
        st.markdown("**💡 Hint : Current Settings**")

        if "atom_type" not in st.session_state:
            st.session_state["atom_type"] = "C3'"
        if "max_dist" not in st.session_state:
            st.session_state["max_dist"] = 20.0
        if "bin_width" not in st.session_state:
            st.session_state["bin_width"] = 1.0

        st.caption(f"Atom: {st.session_state['atom_type']}")
        st.caption(f"Max Dist: {st.session_state['max_dist']} Å")
        st.caption(f"Bin Width: {st.session_state['bin_width']} Å")

    # --- PAGE: WELCOME ---
    if "Welcome" in nav_selection:
        st.title("Welcome Page")

        st.markdown("""
        ### Description
        
        **1. Theory: Inverse Boltzmann Principle**
        The pipeline assumes that frequently observed structural features correspond to low-energy states. We calculate a pseudo-energy ($E$) for base pairs using the formula:
        
        $$
        E(r) = -kT \ln \\left( \\frac{P_{obs}(r)}{P_{ref}(r)} \\right)
        $$
        
        Where $P_{obs}$ is the observed probability of a pair (e.g., A-U) being at distance $r$, and $P_{ref}$ is the reference probability in a "pooled" state (ignoring base identity).
        
        **2. Our Pipeline's Logic**
        * **Training:** The model extracts C3'-C3' distances from your training set. It considers only residues with a sequence separation $\ge$ 4 to capture tertiary interactions rather than local backbone geometry.
        * **Visualisation:** Distance-dependent profiles are generated for all 10 base-pair combinations (AA, AU, GC, etc.).
        * **Scoring:** New structures are scored by summing the potentials of all applicable atom pairs. Negative scores indicate a structure that matches the training distribution (i.e. favourable).
        
        ---
        """)

        st.subheader("Data Deposit")
        st.info(
            "💡 **Tip:** You can drag and drop an entire **folder** containing PDB files directly into the box below."
        )

        uploaded_files = st.file_uploader(
            "Upload Training Dataset",
            accept_multiple_files=True,
            type=["pdb"],
            help="Drag a folder of PDB files here, or select multiple files (Ctrl+A).",
        )

        if uploaded_files:
            if not st.session_state["training_data_dir"]:
                st.session_state["training_data_dir"] = tempfile.mkdtemp(
                    prefix="rna_train_"
                )

            paths = []
            progress_text = st.empty()
            progress_text.text("Saving uploaded files to memory...")

            for uploaded_file in uploaded_files:
                file_path = os.path.join(
                    st.session_state["training_data_dir"], uploaded_file.name
                )
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                paths.append(file_path)

            st.session_state["training_file_count"] = len(
                os.listdir(st.session_state["training_data_dir"])
            )

            progress_text.empty()
            st.success(f"✅ Successfully deposited {len(uploaded_files)} PDB files.")
            st.markdown(
                "👉 **Navigate to 'Pipeline Dashboard' in the sidebar to begin analysis.**"
            )

        elif st.session_state["training_file_count"] > 0:
            st.success(
                f"💡 {st.session_state['training_file_count']} files currently in memory."
            )

    # --- PAGE: PIPELINE DASHBOARD ---
    elif "Pipeline" in nav_selection:
        st.title("Pipeline Dashboard")

        # 1. CONFIGURATION
        with st.expander(
            "⚙️ Pipeline Configuration", expanded=not st.session_state["pipeline_run"]
        ):
            c1, c2, c3 = st.columns(3)
            c1.selectbox(
                "Atom Type", ["C3'", "P", "C4'", "C5'", "O3'"], key="atom_type"
            )
            c2.number_input("Max Distance (Å)", value=20.0, key="max_dist")
            c3.number_input("Bin Width (Å)", value=1.0, step=0.1, key="bin_width")

        # 2. EXECUTION
        col_exec, col_status = st.columns([1, 3])

        with col_exec:
            run_btn = st.button(
                "▶ Run Full Pipeline", type="primary", use_container_width=True
            )

        with col_status:
            if st.session_state["training_file_count"] == 0:
                st.warning(
                    "⚠️ No training data found. Please deposit files in 'Welcome'."
                )

        # RUN PIPELINE LOGIC
        if run_btn and st.session_state["training_file_count"] > 0:
            status_box = st.status("Pipeline Running...", expanded=True)

            # A. Training
            status_box.write("1. Training Model...")
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

                status_box.write("2. Generating Visualisations...")
                status_box.update(
                    label="Pipeline Completed Successfully! ✅",
                    state="complete",
                    expanded=False,
                )
                st.rerun()
            else:
                status_box.update(label="Pipeline Failed :(", state="error")
                st.error(msg)

        st.divider()

        # 3. RESULTS AREA
        if st.session_state["pipeline_run"] and st.session_state["potentials_dir"]:
            tab_viz, tab_score, tab_download = st.tabs(
                ["Potentials Plot", "Score Structure", "Files to Download"]
            )

            # --- TAB: VISUALISATION ---
            with tab_viz:
                # NEW Dropdown for plot type
                plot_choice = st.selectbox(
                    "Select Plot Type", ["Combined Overlay", "Grid View"]
                )

                # Pass choice to engine
                fig, err = run_plotting_engine(
                    st.session_state["potentials_dir"], plot_type=plot_choice
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(err)

            # --- TAB: SCORING ---
            with tab_score:
                st.markdown("### Score New Structures")
                st.write(
                    "Upload a PDB file to calculate its pseudo-energy using the model trained above."
                )

                c_up, c_res = st.columns([1, 1])
                with c_up:
                    target_pdb = st.file_uploader("Upload Target PDB", type="pdb")
                    score_trigger = st.button("Calculate Score")

                with c_res:
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
                            st.metric(
                                "Pseudo-Energy",
                                f"{score:.4f}",
                                help="Lower is more favourable",
                            )
                            st.metric("Interactions Used", pairs)

                            if score < -10:
                                st.success("Result: Favourable conformation")
                            elif score > 5:
                                st.error("Result: Unfavourable conformation")
                            else:
                                st.warning("Result: Neutral conformation")

            # --- TAB: DOWNLOADS ---
            with tab_download:
                st.write("Download the trained statistical potentials.")

                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zip_obj:
                    p_dir = st.session_state["potentials_dir"]
                    for f in os.listdir(p_dir):
                        zip_obj.write(os.path.join(p_dir, f), f)
                buf.seek(0)

                st.download_button(
                    label="Download Potentials (.zip)",
                    data=buf,
                    file_name="trained_potentials.zip",
                    mime="application/zip",
                )

        elif not st.session_state["pipeline_run"]:
            st.info(
                "💡 Upload data in 'Welcome', then click 'Run Full Pipeline' above to see results."
            )


if __name__ == "__main__":
    main()
