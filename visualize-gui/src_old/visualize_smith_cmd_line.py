import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Read the CSV file with Greek Windows-1253 encoding
df = pd.read_csv('smith/smith_fixed_batch_results_5x_variation_merged.csv', encoding='cp1253')

# Extract criteria/concepts from the 'Main Answer' column
def extract_concepts(text):
    if pd.isna(text):
        return []
    lines = text.split('\n')
    concepts = []
    for line in lines:
        line = line.strip()
        if line.startswith(tuple(str(i)+'.' for i in range(1, 21))):
            # Markdown or plain numbered list
            if '**' in line:
                c = line.split('**')[1].replace(':', '').replace('.', '').strip()
                if c:
                    concepts.append(c)
            else:
                c = line.lstrip('0123456789. ').replace(':', '').replace('.', '').strip()
                if c:
                    concepts.append(c)
    return concepts

# Add a column for extracted concepts
df['Concepts'] = df['Main Answer'].apply(extract_concepts)

params = ['Temperature', 'Top-p', 'Top-k', 'BM25 Weight']
param_labels = {'Temperature': 'Temp', 'Top-p': 'Topp', 'Top-k': 'Topk', 'BM25 Weight': 'BM25'}

# Ensure output directory exists
outdir = 'smith'
os.makedirs(outdir, exist_ok=True)

# Block definitions: (start_row, end_row, varying_param)
blocks = [
    (0, 5, 'Temperature'),
    (5, 10, 'Top-p'),
    (10, 15, 'Top-k'),
    (15, 20, 'BM25 Weight'),
]

for start, end, varying_param in blocks:
    print(f"\nProcessing block: {varying_param} (rows {start} to {end-1})")
    subset = df.iloc[start:end].copy()
    print(f"Subset shape: {subset.shape}")
    if subset.empty:
        print(f"Block {varying_param} is empty.")
        continue
    print(f"Extracting concepts for block {varying_param}...")
    print(subset[['Main Answer', 'Concepts']])
    mlb = MultiLabelBinarizer()
    concept_matrix = pd.DataFrame(mlb.fit_transform(subset['Concepts']), columns=mlb.classes_)
    print(f"Concept matrix shape: {concept_matrix.shape}")
    for p in params:
        concept_matrix[p] = subset[p].values
    try:
        concept_matrix_reset = concept_matrix.drop(params, axis=1).astype(bool).reset_index(drop=True)
        print(f"Generating upset data for {varying_param}...")
        upset_data = from_indicators(concept_matrix_reset, concept_matrix_reset.columns)
        fig = plt.figure(figsize=(12, 7))
        upset = UpSet(upset_data, show_counts=True)
        axes = upset.plot(fig=fig)
        bar_ax = axes['intersections']
        matrix_ax = axes['matrix']
        bars = bar_ax.patches
        upset_index = upset_data.index  # MultiIndex of intersections

        # Only annotate visible bars (non-zero height), skip first empty column
        for i, (bar, intersection) in enumerate(zip(bars, upset_index)):
            if i == 0:
                continue  # Skip the first column, which is always empty
            if bar.get_height() == 0:
                continue  # Skip invisible bars
            x = bar.get_x() + bar.get_width() / 2
            # Find the parameter value(s) for this intersection
            mask = np.ones(len(concept_matrix), dtype=bool)
            # Use (i-1) to get the correct intersection since we skipped i=0
            intersection_idx = i - 1
            actual_intersection = upset_index[intersection_idx]
            for col, present in zip(concept_matrix.drop(params, axis=1).columns, actual_intersection):
                if present:
                    mask &= concept_matrix[col] == 1
                else:
                    mask &= concept_matrix[col] == 0
            param_vals = concept_matrix.loc[mask, varying_param].unique()
            label = ','.join(map(str, param_vals)) if len(param_vals) > 0 else ''
            # Find the y-position of the topmost filled circle in this column
            present_indices = [j for j, present in enumerate(actual_intersection) if present]
            if not present_indices:
                continue  # Skip empty columns (no filled circles)
            # Place all labels at a consistent height just above the matrix
            y_label = len(concept_matrix_reset.columns) - 0.5  # Just above the top of the matrix
            print(f"DEBUG: Label for col {i}: x={x}, y={y_label}, label={label}")
            matrix_ax.text(x, y_label, label, ha='center', va='bottom', fontsize=10, color='red', rotation=0, clip_on=False)
        # Add fixed values for other parameters in the title
        other_params = [p for p in params if p != varying_param]
        fixed_vals = {param_labels[p]: subset[p].iloc[0] for p in other_params}
        fixed_str = ' | '.join(f"{p}={v}" for p, v in fixed_vals.items())
        plt.suptitle(f"UpSet Diagram: {param_labels[varying_param]} sweep\nOther Params: {fixed_str}", fontsize=14)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        outpath = f"{outdir}/upset_{param_labels[varying_param]}_composed.png"
        print(f"Saving diagram to {outpath} ...")
        plt.savefig(outpath)
        plt.close('all')
        print(f"Diagram saved for {varying_param} block.")
    except Exception as e:
        print(f"Could not generate composed upset plot for {varying_param}: {e}")
        plt.close('all')