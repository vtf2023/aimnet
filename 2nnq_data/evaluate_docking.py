import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

"""
Docking evaluation utility
==========================

Features
--------
✓ Auto-detect existing energy files (no crash on missing ligands)
✓ Compute ROC + AUC
✓ Compute EF1 / EF5 / EF10
✓ Save publication-quality ROC figure
✓ Print top-ranked ligands

Usage example
-------------
python evaluate_docking.py

Just modify ACTIVE_DIR / DECOY_DIR below if needed.
"""

# =====================
# USER SETTINGS
# =====================
ACTIVE_DIR = "./workdir_fam_actives_dxdydz20_compal"
DECOY_DIR = "./workdir_fam_decoys_dxdydz20_compal"
TOP_PERCENTS = [1]
OUTPUT_FIG = "roc_curve.png"


# =====================
# Helpers
# =====================

def read_min_energy(filepath):
    with open(filepath) as f:
        vals = [float(x.strip()) for x in f]
    return np.min(vals)


def collect_energy_files(base_dir, label):
    """
    Automatically detect all ligand_*_match_energies.txt
    Returns energies + labels
    """
    energy_dir = os.path.join(base_dir, "Energy")

    if not os.path.exists(energy_dir):
        raise RuntimeError(f"Energy directory not found: {energy_dir}")

    pattern = r"ligand_(\d+)_match_energies\.txt"

    files = []
    for f in os.listdir(energy_dir):
        if re.match(pattern, f):
            files.append(f)

    if not files:
        raise RuntimeError(f"No energy files found in {energy_dir}")

    # sort by ligand index for clean order
    files.sort(key=lambda x: int(re.search(r"(\d+)", x).group(1)))

    energies = []
    labels = []

    for f in files:
        path = os.path.join(energy_dir, f)
        energies.append(read_min_energy(path))
        labels.append(label)

    return energies, labels, files


# =====================
# Main evaluation
# =====================

def evaluate():
    energies_a, labels_a, files_a = collect_energy_files(ACTIVE_DIR, 1)
    energies_d, labels_d, files_d = collect_energy_files(DECOY_DIR, 0)

    energies = np.array(energies_a + energies_d)
    labels = np.array(labels_a + labels_d)
    file_names = files_a + files_d

    print("=" * 60)
    print(f"Actives : {len(energies_a)}")
    print(f"Decoys  : {len(energies_d)}")
    print(f"Total   : {len(energies)}")
    print("=" * 60)

    # -----------------
    # ROC + AUC
    # -----------------
    fpr, tpr, _ = roc_curve(labels, -energies)
    auc_score = auc(fpr, tpr)

    print(f"AUC = {auc_score:.4f}\n")

    # -----------------
    # Plot ROC
    # -----------------
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Docking ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300)
    plt.show()

    print(f"ROC figure saved → {OUTPUT_FIG}\n")

    # -----------------
    # Enrichment Factors
    # -----------------
    total = len(labels)
    total_actives = np.sum(labels)

    order = np.argsort(energies)  # lower energy is better

    for p in TOP_PERCENTS:
        top_n = max(1, int(total * p / 100))
        top_idx = order[:top_n]
        actives_in_top = np.sum(labels[top_idx])

        ef = (actives_in_top / total_actives) * (total / top_n)
        print(f"EF{p:<2} = {ef:.3f}   (top {top_n} ligands, {actives_in_top} actives)")

    # -----------------
    # Top ranked ligands
    # -----------------
    print("\nTop 20 ligands (best energies):")
    print("index   energy     label")
    for idx in order[:50]:
        print(f"{file_names[idx]:<25} {energies[idx]:8.3f}   {labels[idx]}")


if __name__ == "__main__":
    evaluate()

