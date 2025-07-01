import os
import torch

# Carica la configurazione e imposta le variabili globali
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

RESULTS_FOLDER = "../NETWORKS"
CSV_FILE_BEST_CANDIDATES = os.path.join(RESULTS_FOLDER, "results_best_candidates.csv")
CSV_FILE_ALL_CANDIDATES = os.path.join(RESULTS_FOLDER, "results_all_candidates.csv")
BACKUP_FOLDER = os.path.join(RESULTS_FOLDER, "BACKUP")