import streamlit as st
import os
import math
import tempfile
import shutil
import zipfile
import io
import plotly.graph_objects as go
import plotly.express as px

# Import shared utilities
try:
    from rna_utils import parse_pdb_atoms, get_bin_index, pair_key, load_params, PAIR_TYPES
except ImportError:
    st.error("Error: 'rna_utils.py' not found. Please ensure it is in the same directory.")
    st.stop()

# --- Configuration ---
st.set_page_config(
    page_title="RNA Potentials Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Clean Scientific Look ---
st.markdown("""
    <style>
    /* Clean metrics */
    .stMetric {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Hide default streamlit menu for cleaner look (optional) */
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

def init_session_state():
    if "training_data_dir" not in st.session_state:
        st.session_state["training_data_dir"] = None
    if "potentials_dir" not in st.session_state:
        st.session_state["potentials_dir"] = None
    if "training_file_count" not in st.session_state:
        st.session_state["training_file_count"] = 0

def save_uploaded_files(uploaded_files, target_dir):
    """Saves Streamlit uploaded files to a temporary directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(target_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        paths.append(file_path)
    return paths

def zip_directory(folder_path):
    """Zips a directory into an in-memory bytes buffer."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zip_obj:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zip_obj.write(os.path.join(root, file), 
                              os.path.relpath(os.path.join(root, file), os.path.join(folder_path, '..')))
    buf.seek(0)
    return buf

# --- Core Logic Functions ---

def run_training(pdb_dir, out_dir, atom_type, max_dist, bin_width):
    nbins = int(math.ceil(max_dist / bin_width))
    pair_counts = {p: [0] * nbins for p in PAIR_TYPES}
    ref_counts = [0] * nbins

    files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    
    if not files:
        return False, "No PDB files found."

    status_container = st.status("Processing structures...", expanded=True)
    progress_bar = status_container.progress(0)
    
    processed_count = 0
    total_files = len(files)

    for idx, fpath in enumerate(files):
        if idx % 5 == 0:
            status_container.write(f"Parsing: {os.path.basename(fpath)}")
        
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
                        res_i = residues[i][1]
                        res_j = residues[j][1]
                        key = pair_key(res_i, res_j)
                        if key in pair_counts:
                            pair_counts[key][bin_idx] += 1
                            ref_counts[bin_idx] += 1
        processed_count += 1
        progress_bar.progress((idx + 1) / total_files)

    status_container.update(label="Calculating statistical potentials...", state="running")
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

    with open(os.path.join(out_dir, "params.txt"), "w") as p_f:
        p_f.write(f"{atom_type}\n{max_dist}\n{bin_width}\n")

    summary = f"Files processed: {processed_count}\n"
    summary += f"Total distance counts: {sum(ref_counts)}\n"
    summary += f"Parameters: Atom={atom_type}, Max={max_dist}, Width={bin_width}\n"
    
    with open(os.path.join(out_dir, "summary.txt"), "w") as s_f:
        s_f.write(summary)

    status_container.update(label="Training complete!", state="complete", expanded=False)
    return True, summary

def run_plotting_interactive(pot_dir):
    try:
        atom, max_dist, bin_width = load_params(pot_dir)
    except Exception as e:
        return None, f"Could not load params: {e}"

    nbins = int(math.ceil(max_dist / bin_width))
    x_axis = [i * bin_width + (bin_width / 2) for i in range(nbins)]
    fig = go.Figure()
    colors = px.colors.qualitative.Safe 

    for idx, pair in enumerate(PAIR_TYPES):
        fname = os.path.join(pot_dir, f"potential_{pair}.txt")
        if os.path.exists(fname):
            with open(fname, "r") as f:
                scores = [float(line.strip()) for line in f if line.strip()]
            
            if len(scores) == len(x_axis):
                fig.add_trace(go.Scatter(
                    x=x_axis, y=scores, mode='lines', name=pair,
                    line=dict(width=2, color=colors[idx % len(colors)]),
                    hovertemplate=f"<b>{pair}</b><br>Dist: %{{x:.1f}}Å<br>Energy: %{{y:.2f}}<extra></extra>"
                ))

    fig.update_layout(
        title=f"RNA Statistical Potentials ({atom})",
        xaxis_title="Distance (Å)",
        yaxis_title="Pseudo-energy (kT)",
        template="simple_white",
        hovermode="x unified",
        height=600
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=max_dist, y1=0,
                  line=dict(color="black", width=1, dash="dash"))
    fig.update_yaxes(range=[-11, 11])
    return fig, None

def interpolate_score(dist, scores, bin_width, max_dist):
    if dist <= 0.0: return scores[0]
    if dist >= max_dist: return scores[-1]
    val_idx = dist / bin_width - 0.5
    idx_low = int(math.floor(val_idx))
    idx_high = idx_low + 1
    if idx_low <= 0: return scores[0]
    if idx_high >= len(scores): return scores[-1]
    frac = val_idx - idx_low
    return scores[idx_low] + (scores[idx_high] - scores[idx_low]) * frac

def run_scoring(pdb_path, pot_dir):
    try:
        atom_type, max_dist, bin_width = load_params(pot_dir)
    except:
        return None, None, "Failed to load params.txt."

    potentials = {}
    for pair in PAIR_TYPES:
        fname = os.path.join(pot_dir, f"potential_{pair}.txt")
        if os.path.exists(fname):
            with open(fname, "r") as f:
                potentials[pair] = [float(line.strip()) for line in f if line.strip()]

    chains = parse_pdb_atoms(pdb_path, atom_type)
    if not chains:
        return 0.0, 0, f"No valid {atom_type} atoms found."

    total_score = 0.0
    pairs_used = 0

    for chain_id, residues in chains.items():
        n = len(residues)
        for i in range(n):
            for j in range(i + 4, n):
                r1_name = residues[i][1]
                r1_coords = residues[i][2]
                r2_name = residues[j][1]
                r2_coords = residues[j][2]
                d = math.dist(r1_coords, r2_coords)

                if d < max_dist:
                    key = pair_key(r1_name, r2_name)
                    if key in potentials:
                        score = interpolate_score(d, potentials[key], bin_width, max_dist)
                        total_score += score
                        pairs_used += 1
                        
    return total_score, pairs_used, None

# --- Main App Interface ---

def main():
    init_session_state()

    # --- Sidebar Layout ---
    with st.sidebar:
        st.image("logo.png", use_container_width=True)
        st.markdown("## **RNA Potentials**") 
        st.write("v1.2.0")
        
        st.divider()
        
        # Navigation
        page = st.radio("Navigation", ["Welcome", "Analysis"])

    # --- PAGE: WELCOME ---
    if page == "Welcome":
        st.title("Welcome to RNA Potentials")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Introduction
            This application allows you to derive and apply distance-dependent statistical potentials for RNA structures.
            
            **Methodology:**
            Pseudo-energies are calculated based on the inverse Boltzmann principle using observed frequencies of atom-pair distances in a training dataset.
            
            **Instructions:**
            1. **Deposit Files:** Upload your PDB training dataset below.
            2. **Go to Analysis:** Switch to the Analysis page to train the model, visualize potentials, and score new structures.
            """)
            
        with col2:
            st.info("""
            **Supported Format:** PDB
            **Default Atom:** C3'
            **License:** MIT
            """)

        st.divider()
        
        st.subheader("Data Deposit")
        st.write("Upload the PDB files you wish to use for training the statistical potential.")
        
        uploaded_files = st.file_uploader("Drop PDB files here", accept_multiple_files=True, type=["pdb"])
        
        if uploaded_files:
            # Create a persistent temp dir for this session if not exists
            if not st.session_state["training_data_dir"]:
                st.session_state["training_data_dir"] = tempfile.mkdtemp(prefix="rna_train_")
            
            # Save files
            save_uploaded_files(uploaded_files, st.session_state["training_data_dir"])
            st.session_state["training_file_count"] = len(os.listdir(st.session_state["training_data_dir"]))
            
            st.success(f"Successfully deposited {len(uploaded_files)} files.")
            st.markdown("👉 **Navigate to the 'Analysis' page to proceed.**")

        # Show current status if files exist
        elif st.session_state["training_file_count"] > 0:
            st.info(f"Current Deposit: {st.session_state['training_file_count']} files ready for analysis.")

    # --- PAGE: ANALYSIS ---
    elif page == "Analysis":
        st.title("Analysis Dashboard")

        # Tabs
        tab_train, tab_viz, tab_score = st.tabs(["Train Model", "Visualise Potentials", "Score Structure"])

        # --- TAB: TRAIN ---
        with tab_train:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Settings")
                
                # Atom Type Dropdown
                atom_options = ["C3'", "P", "C4'", "C5'", "O3'", "O5'", "C1'", "C2'"]
                atom_type = st.selectbox("Atom Type", atom_options, index=0, help="Select the atom used for distance calculations")
                
                max_dist = st.number_input("Max Distance (Å)", value=20.0, step=1.0)
                bin_width = st.number_input("Bin Width (Å)", value=1.0, step=0.1)
            
            with col2:
                st.subheader("Execution")
                
                file_count = st.session_state["training_file_count"]
                
                if file_count > 0:
                    st.success(f"Using {file_count} files from Data Deposit.")
                    if st.button("Start Training", type="primary"):
                        with tempfile.TemporaryDirectory() as tmp_out:
                            success, msg = run_training(
                                st.session_state["training_data_dir"], 
                                tmp_out, 
                                atom_type, 
                                max_dist, 
                                bin_width
                            )
                            
                            if success:
                                st.success("Training Complete")
                                
                                # Persist results
                                persist_dir = os.path.join(tempfile.gettempdir(), "streamlit_rna_potentials_results")
                                if os.path.exists(persist_dir): shutil.rmtree(persist_dir)
                                shutil.copytree(tmp_out, persist_dir)
                                st.session_state["potentials_dir"] = persist_dir
                                
                                # Download
                                zip_bytes = zip_directory(tmp_out)
                                st.download_button("Download Potentials (.zip)", zip_bytes, "potentials.zip", "application/zip")
                            else:
                                st.error(msg)
                else:
                    st.warning("No training files found. Please go to the 'Welcome' page to deposit files.")

        # --- TAB: VISUALISE ---
        with tab_viz:
            active_dir = st.session_state["potentials_dir"]
            
            # Allow external upload override if needed, essentially "load from zip"
            col_opt, _ = st.columns([1, 2])
            with col_opt:
                upload_override = st.file_uploader("Or load external potentials (.zip)", type="zip")
            
            if upload_override:
                temp_extract = os.path.join(tempfile.gettempdir(), "streamlit_uploaded_pots")
                if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
                os.makedirs(temp_extract)
                with zipfile.ZipFile(upload_override, "r") as z:
                    z.extractall(temp_extract)
                # handle nesting
                if not os.path.exists(os.path.join(temp_extract, "params.txt")):
                    subdirs = [d for d in os.listdir(temp_extract) if os.path.isdir(os.path.join(temp_extract, d))]
                    if subdirs: temp_extract = os.path.join(temp_extract, subdirs[0])
                active_dir = temp_extract

            if active_dir:
                fig, err = run_plotting_interactive(active_dir)
                if err:
                    st.error(err)
                else:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Train a model or upload a zip to see visualizations.")

        # --- TAB: SCORE ---
        with tab_score:
            col_in, col_res = st.columns(2)
            
            with col_in:
                st.subheader("Input")
                target_pdb = st.file_uploader("Target PDB Structure", type="pdb")
                pot_dir = st.session_state["potentials_dir"] or (active_dir if 'active_dir' in locals() else None)
                
                if not pot_dir:
                    st.error("No potentials loaded.")
                    
                score_btn = st.button("Calculate Score", disabled=(not pot_dir or not target_pdb))

            with col_res:
                st.subheader("Results")
                if score_btn and pot_dir and target_pdb:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
                        tmp.write(target_pdb.getbuffer())
                        tpath = tmp.name
                    
                    score, pairs, err = run_scoring(tpath, pot_dir)
                    
                    if err:
                        st.error(err)
                    else:
                        st.metric("Total Pseudo-Energy", f"{score:.4f}")
                        st.metric("Interactions Counted", pairs)
                        
                        if score < 0:
                            st.success("Favourable conformation")
                        else:
                            st.warning("Unfavourable conformation")

if __name__ == "__main__":
    main()