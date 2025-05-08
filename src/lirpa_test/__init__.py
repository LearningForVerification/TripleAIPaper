import torch

from .config import (
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
    CSV_FILE_BEST_CANDIDATES,
    CSV_FILE_ALL_CANDIDATES,
    BACKUP_FOLDER
)

device = "cuda" if torch.cuda.is_available() else "cpu"
