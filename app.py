import streamlit as st
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
import io
from collections import Counter
from math import factorial

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

        step = 10
        window = 10
        results = []

        for i in range(0, n_atoms - window, step):
            ts = [np.linalg.norm(ca_coords[i] - ca_coords[j]) for j in range(i + 1, i + window + 1)]
            ts = np.array(ts)
            diff_series = np.abs(np.diff(ts))
            diff_series = diff_series[diff_series > 0]
            lyap = np.mean(np.log(diff_series)) if len(diff_series) > 0 else 0
            rr, det, entr = compute_rqa(ts)
            pe = permutation_entropy(ts)
            results.append({
                "Start": i,
                "End": i + window,
                "Lyapunov": round(lyap, 4),
                "RQA_RR": round(rr, 4),
                "RQA_DET": round(det, 4),
                "RQA_ENTR": round(entr, 4),
                "PE": round(pe, 4),
                "IDR_overlap": "Yes" if i <= 60 <= (i + window) else "No"
            })

        st.markdown("### 📊 Analysis Summary Table")
        st.dataframe(results)

        # CSV download
        csv_output = io.StringIO()
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(csv_output, index=False)
        st.download_button("⬇️ Download CSV Results", data=csv_output.getvalue(), file_name="chaos_analysis_summary.csv")
