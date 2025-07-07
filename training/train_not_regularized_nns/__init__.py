import os

import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory NETWORKS, che si trova una cartella sopra BASE_DIR
RESULTS_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'networks'))
print(f"RESULTS_FOLDER: {RESULTS_FOLDER}")

# Ora i path assoluti usando RESULTS_FOLDER
BEST_MODELS_FOLDER = os.path.join(RESULTS_FOLDER, "MNIST", "best_models")
print(f"BEST_MODELS_FOLDER: {BEST_MODELS_FOLDER}")
BACKUP_FOLDER = os.path.join(RESULTS_FOLDER, "BACKUP")
print(f"BACKUP_FOLDER: {BACKUP_FOLDER}")
ALL_MODELS_FOLDER = os.path.join(RESULTS_FOLDER, "MNIST", "all_models")
print(f"ALL_MODELS_FOLDER: {ALL_MODELS_FOLDER}")
REFINED_MODELS_FOLDER = os.path.join(RESULTS_FOLDER, "MNIST", "refined_attempt")
print(f"REFINED_MODELS_FOLDER: {REFINED_MODELS_FOLDER}")

CSV_FILE_BEST_CANDIDATES = os.path.join(RESULTS_FOLDER, "results_best_candidates.csv")
print(f"CSV_FILE_BEST_CANDIDATES: {CSV_FILE_BEST_CANDIDATES}")
CSV_FILE_ALL_CANDIDATES = os.path.join(RESULTS_FOLDER, "results_all_candidates.csv")
print(f"CSV_FILE_ALL_CANDIDATES: {CSV_FILE_ALL_CANDIDATES}")
# Creazione cartelle se non esistono
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(BEST_MODELS_FOLDER, exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)
os.makedirs(ALL_MODELS_FOLDER, exist_ok=True)
os.makedirs(REFINED_MODELS_FOLDER, exist_ok=True)


from config import (
    REFINEMENT_CYCLE_LENGTH,
    NUMBER_OF_CYCLES,
    REFINEMENT_PERCENTAGE,
    NUM_EPOCHS,
    NOISE,
    ACCURACY_THRESHOLD,
    RS_LOSS_FIRST_NN,
    VALIDATION_FREQUENCY,
    DEBUG,
    RESULTS_FOLDER,
    LAMBDA_LR_CYCLE,
    CONV_RSLOSS_LAMBDA_MULTIPLIER,
    FIXED_LR,
    LR_DECAY
)

device = "cuda" if torch.cuda.is_available() else "cpu"
