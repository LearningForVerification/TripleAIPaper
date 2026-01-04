import os
import torch

# Carica la configurazione e imposta le variabili globali
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

DATASET_NAME = "CIFAR10"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory NETWORKS, che si trova una cartella sopra BASE_DIR
RESULTS_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'networks'))

# Ora i path assoluti usando RESULTS_FOLDER
BEST_MODELS_FOLDER = os.path.join(RESULTS_FOLDER, DATASET_NAME, "best_models")
BACKUP_FOLDER = os.path.join(RESULTS_FOLDER, "BACKUP")
ALL_MODELS_FOLDER = os.path.join(RESULTS_FOLDER, DATASET_NAME, "all_models")
REFINED_MODELS_FOLDER = os.path.join(RESULTS_FOLDER, DATASET_NAME, "refined_attempt")


CSV_FILE_BEST_CANDIDATES = os.path.join(RESULTS_FOLDER, "results_best_candidates.csv")
CSV_FILE_ALL_CANDIDATES = os.path.join(RESULTS_FOLDER, "results_all_candidates.csv")

# Creazione cartelle se non esistono
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(BEST_MODELS_FOLDER, exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)
os.makedirs(ALL_MODELS_FOLDER, exist_ok=True)
os.makedirs(REFINED_MODELS_FOLDER, exist_ok=True)
