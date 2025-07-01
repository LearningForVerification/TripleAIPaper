import configparser
import os

# Percorso del file .ini
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)

# Parsing con corretta tipizzazione
REFINEMENT_CYCLE_LENGTH = config.getint('DEFAULT', 'refinement_cycle_length')
NUMBER_OF_CYCLES = config.getint('DEFAULT', 'number_of_cycles')
REFINEMENT_PERCENTAGE = config.getfloat('DEFAULT', 'refinement_percentage')

NUM_EPOCHS = config.getint('DEFAULT', 'num_epochs')
NOISE = config.getfloat('DEFAULT', 'noise')
ACCURACY_THRESHOLD = config.getfloat('DEFAULT', 'accuracy_threshold')

RS_LOSS_FIRST_NN = config.getfloat('DEFAULT', 'rs_loss_first_nn')
VALIDATION_FREQUENCY = config.getint('DEFAULT', 'validation_frequency')

RESULTS_FOLDER = "NETWORKS"
CSV_FILE_BEST_CANDIDATES = os.path.join(RESULTS_FOLDER, "results_best_candidates.csv")
CSV_FILE_ALL_CANDIDATES = os.path.join(RESULTS_FOLDER, "results_all_candidates.csv")
BACKUP_FOLDER = os.path.join(RESULTS_FOLDER, "BACKUP")
LAMBDA_LR_CYCLE = config.getint('DEFAULT', 'lambda_lr_cycle')
CONV_RSLOSS_LAMBDA_MULTIPLIER = config.getfloat('DEFAULT', 'conv_rsloss_lambda_multiplier')
FIXED_LR = config.getboolean('DEFAULT', 'fixed_lr')
LR_DECAY = config.getfloat('DEFAULT', 'lr_decay')
DEBUG = config.getboolean('DEFAULT', 'debug')