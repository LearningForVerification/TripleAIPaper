import os, sys
import csv
import logging
# Add parent directory and current directory to the system path
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
sys.path.insert(0, parent_directory)
from verificatori.marabou.get_marabou_time import get_marabou_time

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger()

# Directory corrente
current_directory = os.path.dirname(os.path.abspath(__file__))

# Cartelle delle categorie di esperimenti
experiments_category_folders = ["2-FC", "CONV", "FC"]
experiments_category_folders = [os.path.join(current_directory, "networks", x) for x in experiments_category_folders]

# Sottocategorie
sub_category_folder = ["0.03", "not_over_param", "over_param"]

# Cartella proprietà (deve esistere)
property_folder = os.path.join(current_directory, "properties", "0.03")
if not os.path.isdir(property_folder):
    raise Exception(f"Directory '{property_folder}' not found")

# Controlla esistenza delle directory principali e delle sottocartelle
for folder in experiments_category_folders:
    if not os.path.isdir(folder):
        raise Exception(f"Directory '{folder}' not found")

    for sub_folder in sub_category_folder:
        sub_path = os.path.join(folder, sub_folder)
        if not os.path.isdir(sub_path):
            raise Exception(f"Directory '{sub_path}' not found")

# Crea cartelle in 'results/<categoria>' e file CSV vuoti per ciascuna sottocategoria
results_base = os.path.join(current_directory, "results")
os.makedirs(results_base, exist_ok=True)

for folder in experiments_category_folders:
    category_name = os.path.basename(folder)
    result_category_path = os.path.join(results_base, category_name)
    os.makedirs(result_category_path, exist_ok=True)

    for sub_folder in sub_category_folder:
        csv_path = os.path.join(result_category_path, f"{sub_folder}.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model path", "property path", "status", "time"])

max_prop = 1 # Limite proprietà

for folder in experiments_category_folders:
    category_name = os.path.basename(folder)
    for sub_folder in sub_category_folder:
        sub_path = os.path.join(folder, sub_folder)
        result_csv_path = os.path.join(results_base, category_name, f"{sub_folder}.csv")

        with open(result_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)

            for nn_file in sorted(os.listdir(sub_path)):
                if not nn_file.endswith(".onnx"):
                    continue

                nn_path = os.path.join(sub_path, nn_file)
                logger.info(f"➡️ Valutazione rete: {nn_file}")

                prop_files = sorted(os.listdir(property_folder))[:max_prop]

                for i, prop_file in enumerate(prop_files, start=1):
                    prop_path = os.path.join(property_folder, prop_file)
                    logger.info(f"   └─ Proprietà {i}/{max_prop}: {prop_file}")

                    elapsed, status = get_marabou_time(nn_path, prop_path, 5)

                    writer.writerow([nn_file, prop_file, status, elapsed])

                logger.info(f"✅ Completata rete: {nn_file}")
