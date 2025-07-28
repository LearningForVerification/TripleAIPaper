import os
import sys
import csv
import logging
import argparse

# Import da script esterni
from get_pynever_time import get_pynever_time

def star_exp(max_prop, timeout):
    # Configura il logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    logger = logging.getLogger()

    # Directory corrente
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Cartelle delle categorie di esperimenti
    experiments_category_folders = ["2-FC", "FC"]
    experiments_category_folders = [os.path.join(current_directory, "networks", x) for x in
                                    experiments_category_folders]

    # Sottocategorie
    sub_category_folder = ["0.03/flatten", "not_over_param/flatten", "over_param/flatten", "not_over_param_not_sparse/flatten"]

    # Cartella proprietà
    property_folder = os.path.join(current_directory, "properties", "0.03")
    if not os.path.isdir(property_folder):
        raise Exception(f"Directory '{property_folder}' not found")

    # Controlla esistenza delle directory
    for folder in experiments_category_folders:
        if not os.path.isdir(folder):
            raise Exception(f"Directory '{folder}' not found")

        for sub_folder in sub_category_folder:
            sub_path = os.path.join(folder, sub_folder)
            if not os.path.isdir(sub_path):
                raise Exception(f"Directory '{sub_path}' not found")

    # Crea cartelle dei risultati e file CSV vuoti
    results_base = os.path.join(current_directory, "results")
    os.makedirs(results_base, exist_ok=True)

    for folder in experiments_category_folders:
        category_name = os.path.basename(folder)
        result_category_path = os.path.join(results_base, category_name)
        os.makedirs(result_category_path, exist_ok=True)

        for sub_folder in sub_category_folder:
            csv_path = os.path.join(result_category_path, f"{sub_folder}.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["model path", "property path", "status", "time"])

    # Loop su tutte le reti e proprietà
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

                        status, elapsed = get_pynever_time(nn_path, prop_path, timeout=timeout)
                        writer.writerow([nn_file, prop_file, status, elapsed])

                    logger.info(f"✅ Completata rete: {nn_file}")


def main():
    parser = argparse.ArgumentParser(description="Verifica proprietà su reti neurali usando PyNEVer.")
    parser.add_argument("--max_prop", type=int, default=1, help="Numero massimo di proprietà da verificare per ogni rete")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout per la verifica (in secondi)")
    args = parser.parse_args()

    star_exp(max_prop=args.max_prop, timeout=args.timeout)

def e_test():
    get_pynever_time(model_path="networks/FC/over_param/fcnn_30.onnx", property_path="properties/0.03/sample_0095_label_4_eps_0.030.vnnlib", timeout=15)


if __name__ == "__main__":
    main()
