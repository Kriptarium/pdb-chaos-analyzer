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
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", uploaded_file)

    # Extract C-alpha atoms
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].get_coord())

    ca_coords = np.array(ca_coords)
    if len(ca_coords) < 20:
        st.warning("The structure has too few Cα atoms for a meaningful analysis.")
    else:
        # Time series: distance between fixed atom 0 and others
        time_series = [np.linalg.norm(ca_coords[0] - ca_coords[i]) for i in range(1, len(ca_coords))]
        ts = np.array(time_series)

        # Simple Lyapunov estimate using finite differences (proxy for demonstration)
        diff_series = np.abs(np.diff(ts))
        diff_series = diff_series[diff_series > 0]  # Avoid log(0)
        lyapunov_est = np.mean(np.log(diff_series))

        st.markdown(f"### 🧮 Estimated Lyapunov Exponent: `{lyapunov_est:.4f}`")

        # Plot the time series
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ts, marker='o', linestyle='-', color='teal')
        ax.set_title("Distance Time Series (Cα Atom 0 vs. Others)")
        ax.set_xlabel("Residue Index")
        ax.set_ylabel("Distance (Å)")
        ax.grid(True)
        st.pyplot(fig)

        # Downloadable data
        csv = io.StringIO()
        np.savetxt(csv, ts, delimiter=",")
        st.download_button("⬇️ Download Time Series Data", data=csv.getvalue(), file_name="ca_distances.csv")
