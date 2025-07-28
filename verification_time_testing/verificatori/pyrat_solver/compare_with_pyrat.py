import logging
import subprocess
import time
import argparse
import os
import csv
import re

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger()



def run_pyrat(model_path: str, property_path: str, timeout: int ):
    cmd = [
        "python", "pyrat.pyc",
        "--model_path", model_path,
        "--property_path", property_path,
        "--split", "True",
        "--verbose", "True",
        "--nb_process", "4",
        "--domains", "zonotopes",
        "--timeout", str(timeout)
    ]

    print(f"Eseguo: modello={model_path}, proprietà={property_path}")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout

        match = re.search(r"Result\s*=\s*(True|False)", output)
        if match:
            if match.group(1) == "True":
                status = "verified"
            else:
                status = "not_verified"
        else:
            status = "error"

    except subprocess.TimeoutExpired:
        print(f"Timeout dopo {timeout} secondi.")
        status = "error"

    end_time = time.time()
    print(f"Tempo totale: {end_time - start_time:.2f} secondi")
    print(f"Status: {status}")

    elapsed = end_time - start_time




    return elapsed, status




def main():
    parser = argparse.ArgumentParser(description="Esegui Pyrat su tutte le combinazioni modelli-proprietà.")
    parser.add_argument("--timeout", type=int, default=50, help="Timeout in secondi per ogni run")
    parser.add_argument("--max_prop", type=int, default=15, help="Timeout in secondi per ogni run")

    args = parser.parse_args()

    max_prop = args.max_prop
    # Directory corrente
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Cartelle delle categorie di esperimenti
    experiments_category_folders = ["2-FC", "FC"]
    experiments_category_folders = [os.path.join(current_directory, "networks", x) for x in
                                    experiments_category_folders]

    sub_category_folder = ["0.03", "not_over_param", "over_param", "not_over_param_not_sparse"]

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

                        elapsed, status = run_pyrat(nn_path, prop_path, args.timeout)

                        writer.writerow([nn_file, prop_file, status, elapsed])

                    logger.info(f"✅ Completata rete: {nn_file}")

def e_test():
    run_pyrat(r"networks/FC/over_param/fcnn_30.onnx", r"properties/0.03/sample_0046_label_1_eps_0.030.vnnlib", 25)


if __name__ == "__main__":
    main()
