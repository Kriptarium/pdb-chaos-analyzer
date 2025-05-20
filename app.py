import streamlit as st
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="PDB Chaos Analyzer", layout="centered")
st.title("🔬 PDB File Lyapunov Analyzer")
st.markdown("Upload a `.pdb` file to compute a simple Lyapunov exponent based on Cα distances.")

uploaded_file = st.file_uploader("Choose a PDB file", type=["pdb"])

if uploaded_file is not None:
    # Convert uploaded binary file to text stream
    text = uploaded_file.read().decode("utf-8")
    text_stream = io.StringIO(text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", text_stream)

    # Extract C-alpha atoms
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].get_coord())

    ca_coords = np.array(ca_coords)
    n_atoms = len(ca_coords)
    
    if n_atoms < 5:
        st.warning("The structure has too few Cα atoms for a meaningful analysis.")
    else:
        st.markdown(f"### ✅ Loaded {n_atoms} Cα atoms.")

        # Atom pair selection
        col1, col2 = st.columns(2)
        with col1:
            start_index = st.number_input("Select first atom index (0-based)", min_value=0, max_value=n_atoms-2, value=0)
        with col2:
            end_index = st.number_input("Select second atom index (0-based)", min_value=start_index+1, max_value=n_atoms-1, value=start_index+10)

        # Generate time series based on selected atom pair
        time_series = [np.linalg.norm(ca_coords[start_index] - ca_coords[i]) for i in range(start_index + 1, end_index + 1)]
        ts = np.array(time_series)

        # Simple Lyapunov estimate using finite differences
        diff_series = np.abs(np.diff(ts))
        diff_series = diff_series[diff_series > 0]  # Avoid log(0)
        lyapunov_est = np.mean(np.log(diff_series))

        st.markdown(f"### 🧮 Estimated Lyapunov Exponent (Atoms {start_index}–{end_index}): `{lyapunov_est:.4f}`")

        # Plot the time series
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ts, marker='o', linestyle='-', color='teal')
        ax.set_title(f"Distance Time Series (Atom {start_index} vs. {start_index+1} to {end_index})")
        ax.set_xlabel("Relative Residue Index")
        ax.set_ylabel("Distance (Å)")
        ax.grid(True)
        st.pyplot(fig)

        # Downloadable data
        csv = io.StringIO()
        np.savetxt(csv, ts, delimiter=",")
        st.download_button("⬇️ Download Time Series Data", data=csv.getvalue(), file_name="ca_distances.csv")
