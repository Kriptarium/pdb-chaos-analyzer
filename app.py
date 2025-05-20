import streamlit as st
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
import io
from collections import Counter
from math import factorial
import pandas as pd

st.set_page_config(page_title="PDB Chaos Analyzer", layout="centered")
st.title("🔬 PDB File Chaos Analyzer (Multi-Metric)")
st.markdown("Upload a `.pdb` file to compute Lyapunov, RQA, and Permutation Entropy on Cα distances.")

# Permutation Entropy function
def permutation_entropy(time_series, order=3, delay=1):
    n = len(time_series)
    permutations = []
    for i in range(n - delay * (order - 1)):
        window = time_series[i:i + delay * order:delay]
        ranks = tuple(np.argsort(window))
        permutations.append(ranks)
    counter = Counter(permutations)
    total = sum(counter.values())
    probs = np.array([count / total for count in counter.values()])
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(factorial(order))
    return entropy / max_entropy

# RQA metrics approximation
def compute_rqa(ts, eps=0.05):
    N = len(ts)
    matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if abs(ts[i] - ts[j]) < eps:
                matrix[i, j] = True
    rr = np.sum(matrix) / (N * N)
    def diag_lines(matrix, min_length=2):
        N = matrix.shape[0]
        counts = []
        for offset in range(1, N):
            diag = np.diagonal(matrix, offset=offset)
            count = 0
            for val in diag:
                if val:
                    count += 1
                else:
                    if count >= min_length:
                        counts.append(count)
                    count = 0
            if count >= min_length:
                counts.append(count)
        return counts
    lines = diag_lines(matrix)
    if len(lines) == 0:
        det = 0
        entr = 0
    else:
        det = np.sum(lines) / np.sum(matrix)
        probs = np.array(lines) / np.sum(lines)
        entr = -np.sum(probs * np.log(probs))
    return rr, det, entr

uploaded_file = st.file_uploader("Choose a PDB file", type=["pdb"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    text_stream = io.StringIO(text)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", text_stream)

    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].get_coord())

    ca_coords = np.array(ca_coords)
    n_atoms = len(ca_coords)

    if n_atoms < 30:
        st.warning("Structure has too few Cα atoms. At least 30 required.")
    else:
        st.markdown(f"### ✅ Extracted {n_atoms} Cα atoms. Running analysis...")

        # IDR region input from user
        st.sidebar.markdown("### 🔍 IDR Region Settings")
        idr_start = st.sidebar.number_input("IDR Start Index", min_value=0, max_value=n_atoms-2, value=0)
        idr_end = st.sidebar.number_input("IDR End Index", min_value=idr_start+1, max_value=n_atoms-1, value=min(60, n_atoms-1))

        step = 10
        window = 10
        results = []
        starts, lyaps, pes = [], [], []

        for i in range(0, n_atoms - window, step):
            ts = [np.linalg.norm(ca_coords[i] - ca_coords[j]) for j in range(i + 1, i + window + 1)]
            ts = np.array(ts)
            diff_series = np.abs(np.diff(ts))
            diff_series = diff_series[diff_series > 0]
            lyap = np.mean(np.log(diff_series)) if len(diff_series) > 0 else 0
            rr, det, entr = compute_rqa(ts)
            pe = permutation_entropy(ts)
            overlap = "Yes" if (idr_start <= i + window and idr_end >= i) else "No"
            results.append({
                "Start": i,
                "End": i + window,
                "Lyapunov": round(lyap, 4),
                "RQA_RR": round(rr, 4),
                "RQA_DET": round(det, 4),
                "RQA_ENTR": round(entr, 4),
                "PE": round(pe, 4),
                "IDR_overlap": overlap
            })
            starts.append(i)
            lyaps.append(lyap)
            pes.append(pe)

        st.markdown("### 📊 Analysis Summary Table")
        df = pd.DataFrame(results)
        st.dataframe(df)

        # Download CSV
        csv_output = io.StringIO()
        df.to_csv(csv_output, index=False)
        st.download_button("⬇️ Download CSV Results", data=csv_output.getvalue(), file_name="chaos_analysis_summary.csv")

        # Plot Lyapunov and PE as a function of start index
        st.markdown("### 📈 Segment-wise Metric Visualization")
        fig1, ax1 = plt.subplots()
        ax1.plot(starts, lyaps, marker='o', label='Lyapunov', color='crimson')
        ax1.set_xlabel("Start Index")
        ax1.set_ylabel("Lyapunov Exponent")
        ax1.set_title("Lyapunov vs. Segment Start Index")
        ax1.grid(True)
        ax1.legend()
        svg1 = io.StringIO()
        fig1.savefig(svg1, format='svg')
        st.pyplot(fig1)
        st.download_button("⬇️ Download Lyapunov Plot (SVG)", data=svg1.getvalue(), file_name="lyapunov_plot.svg")

        fig2, ax2 = plt.subplots()
        ax2.plot(starts, pes, marker='s', label='Permutation Entropy', color='navy')
        ax2.set_xlabel("Start Index")
        ax2.set_ylabel("PE")
        ax2.set_title("Permutation Entropy vs. Segment Start Index")
        ax2.grid(True)
        ax2.legend()
        svg2 = io.StringIO()
        fig2.savefig(svg2, format='svg')
        st.pyplot(fig2)
        st.download_button("⬇️ Download PE Plot (SVG)", data=svg2.getvalue(), file_name="pe_plot.svg")
