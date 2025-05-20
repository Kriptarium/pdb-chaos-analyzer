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

        # Sidebar settings for IDR and filters
        st.sidebar.markdown("### 🔍 IDR & Filter Settings")
        idr_start = st.sidebar.number_input("IDR Start Index", min_value=0, max_value=n_atoms-2, value=0)
        idr_end = st.sidebar.number_input("IDR End Index", min_value=idr_start+1, max_value=n_atoms-1, value=min(60, n_atoms-1))
        min_lyap, max_lyap = st.sidebar.slider("Lyapunov Range", -5.0, 5.0, (-1.0, 1.0))
        min_pe, max_pe = st.sidebar.slider("Permutation Entropy Range", 0.0, 1.0, (0.0, 1.0))
        show_idr_only = st.sidebar.checkbox("Show Only IDR Overlap Segments", value=False)

        step = 10
        window = 10
        results = []
        starts, lyaps, pes, rqa_rrs, rqa_dets, rqa_ents = [], [], [], [], [], []

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
            rqa_rrs.append(rr)
            rqa_dets.append(det)
            rqa_ents.append(entr)

        df = pd.DataFrame(results)
        filtered_df = df[
            (df['Lyapunov'] >= min_lyap) &
            (df['Lyapunov'] <= max_lyap) &
            (df['PE'] >= min_pe) &
            (df['PE'] <= max_pe)
        ]
        if show_idr_only:
            filtered_df = filtered_df[filtered_df['IDR_overlap'] == "Yes"]

        st.markdown("### 📊 Filtered Analysis Table")
        st.dataframe(filtered_df)

        csv_output = io.StringIO()
        filtered_df.to_csv(csv_output, index=False)
        st.download_button("⬇️ Download Filtered CSV", data=csv_output.getvalue(), file_name="filtered_chaos_analysis.csv")

        # Visualizations remain unchanged and reflect all segments
        st.markdown("### 📈 Metric Visualizations (Full Dataset)")

        fig1, ax1 = plt.subplots()
        ax1.plot(starts, lyaps, marker='o', label='Lyapunov', color='crimson')
        ax1.set_xlabel("Start Index")
        ax1.set_ylabel("Lyapunov Exponent")
        ax1.set_title("Lyapunov vs. Segment Start Index")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(starts, pes, marker='s', label='Permutation Entropy', color='navy')
        ax2.set_xlabel("Start Index")
        ax2.set_ylabel("PE")
        ax2.set_title("Permutation Entropy vs. Segment Start Index")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(starts, rqa_rrs, marker='^', label='Recurrence Rate', color='darkgreen')
        ax3.set_xlabel("Start Index")
        ax3.set_ylabel("RQA RR")
        ax3.set_title("Recurrence Rate vs. Segment Start Index")
        ax3.grid(True)
        ax3.legend()
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots()
        ax4.plot(starts, rqa_dets, marker='x', label='Determinism', color='darkorange')
        ax4.set_xlabel("Start Index")
        ax4.set_ylabel("RQA DET")
        ax4.set_title("Determinism vs. Segment Start Index")
        ax4.grid(True)
        ax4.legend()
        st.pyplot(fig4)

        fig5, ax5 = plt.subplots()
        ax5.plot(starts, rqa_ents, marker='d', label='RQA Entropy', color='purple')
        ax5.set_xlabel("Start Index")
        ax5.set_ylabel("RQA ENTR")
        ax5.set_title("RQA Entropy vs. Segment Start Index")
        ax5.grid(True)
        ax5.legend()
        st.pyplot(fig5)
