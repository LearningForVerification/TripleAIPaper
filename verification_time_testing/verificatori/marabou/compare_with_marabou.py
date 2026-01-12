import argparse
import csv
import logging
import os
import signal
import time
import multiprocessing as mp
import numpy as np
from maraboupy import Marabou

# ---------------------- CONFIGURAZIONE LOG ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

# ---------------------- FUNZIONI ----------------------
def run_solver(onnx_path, vnnlib_path, options, queue):
    """Funzione eseguita in multiprocessing per risolvere una proprietà Marabou"""
    try:
        net = Marabou.read_onnx(onnx_path)
        start = time.time()
        result = net.solve(propertyFilename=vnnlib_path, options=options)
        end = time.time()
        elapsed = end - start
        status = result[0]

        if status == 'unsat':
            queue.put((elapsed, 'verified'))
        elif status == 'sat':
            queue.put((elapsed, 'not_verified'))
        else:
            queue.put((elapsed, status))
    except Exception as e:
        queue.put((0.0, f'error: {str(e)}'))

def get_marabou_time(onnx_path, vnnlib_path, timeout):
    """Verifica una proprietà VNN-LIB su una rete ONNX usando Marabou con timeout robusto"""
    options = Marabou.createOptions(
        verbosity=0,
        snc=True,
        numWorkers=1,
        initialTimeout=2,
        initialSplits=4,
        onlineSplits=8,
        timeoutFactor=1.5,
        splittingStrategy="auto",
        sncSplittingStrategy="largest-interval",
        tighteningStrategy="deeppoly",
        milpTightening="lp",
        dumpBounds=False,
        numSimulations=10,
        numBlasThreads=2,
        performLpTighteningAfterSplit=True,
        lpSolver="native"
    )

    queue = mp.Queue()
    p = mp.Process(target=run_solver, args=(onnx_path, vnnlib_path, options, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        time.sleep(0.5)
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
        p.join()
        return timeout, 'timeout'

    if queue.empty():
        return timeout, 'timeout'

    return queue.get()

# ---------------------- SCRIPT PRINCIPALE ----------------------
def main(max_prop):
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Dataset disponibili
    dataset_names = ["FMNIST", "CIFAR_CUSTOM"]

    # Categorie esperimenti
    categories = ["CONV", "FC", "2-FC"]

    # Dataset disponibili
    dataset_names = ["FMNIST"]

    # Categorie esperimenti
    categories = ["CONV"]

    # Timeout fissi per categoria
    timeout_per_category = {
        "CONV": 400,
        "FC": 15,
        "2-FC": 180
    }



    # Sottocategorie
    sub_category_folder = ["0.03", "over_param"]

    # Creazione cartelle base risultati
    results_base = os.path.join(current_directory, "results")
    os.makedirs(results_base, exist_ok=True)

    # Loop principale: dataset → categoria → sottocategoria
    for dataset_name in dataset_names:
        dataset_path = os.path.join(current_directory, "networks", dataset_name)
        if not os.path.isdir(dataset_path):
            logger.warning(f"⚠️ Dataset '{dataset_name}' non trovato, skipping.")
            continue

        for category_name in categories:
            category_path = os.path.join(dataset_path, category_name)
            if not os.path.isdir(category_path):
                logger.warning(f"⚠️ Categoria '{category_name}' non trovata in dataset '{dataset_name}', skipping.")
                continue

            category_timeout = timeout_per_category.get(category_name, 60)
            dataset_result_path = os.path.join(results_base, category_name, dataset_name)
            os.makedirs(dataset_result_path, exist_ok=True)

            # Cartella proprietà
            property_folder = os.path.join(current_directory, "properties", dataset_name, "0.03")
            if not os.path.isdir(property_folder):
                logger.warning(f"⚠️ Cartella proprietà '{property_folder}' non trovata, skipping dataset '{dataset_name}' categoria '{category_name}'.")
                continue

            for sub_folder in sub_category_folder:
                sub_path = os.path.join(category_path, sub_folder)
                if not os.path.isdir(sub_path):
                    logger.warning(f"⚠️ Subfolder '{sub_folder}' non trovato per dataset '{dataset_name}' in categoria '{category_name}', skipping.")
                    continue

                result_csv_path = os.path.join(dataset_result_path, f"{sub_folder}.csv")

                # Creo CSV con header
                with open(result_csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["model path", "property path", "status", "time"])

                # Apro CSV in append
                with open(result_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)

                    for nn_file in sorted(os.listdir(sub_path)):
                        if not nn_file.endswith(".onnx"):
                            continue

                        nn_path = os.path.join(sub_path, nn_file)
                        logger.info(f"➡️ Valutazione rete: {nn_file} con timeout={category_timeout}s")

                        prop_files = sorted(os.listdir(property_folder))[:max_prop]

                        # Statistiche per riga riassuntiva
                        times_for_median = []
                        num_timeout = 0
                        num_failure = 0

                        for i, prop_file in enumerate(prop_files, start=1):
                            prop_path = os.path.join(property_folder, prop_file)
                            logger.info(f"   └─ Proprietà {i}/{max_prop}: {prop_file}")

                            elapsed, status = get_marabou_time(nn_path, prop_path, timeout=category_timeout)

                            # Tutte le proprietà non timeout contribuiscono alla mediana
                            if status != "timeout":
                                times_for_median.append(elapsed)

                            if status == "timeout":
                                num_timeout += 1
                            elif status not in ["verified", "not_verified"]:
                                num_failure += 1

                            writer.writerow([nn_file, prop_file, status, elapsed])

                        # Riga riassuntiva per la rete
                        median_time = np.median(times_for_median) if times_for_median else 0.0
                        writer.writerow([nn_file, "SUMMARY", f"median_time={median_time:.2f}s, timeouts={num_timeout}, failures={num_failure}", ""])
                        logger.info(f"✅ Completata rete: {nn_file} | median_time={median_time:.2f}s, timeouts={num_timeout}, failures={num_failure}")


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valutazione reti con Marabou e timeout per proprietà VNN-LIB.")
    parser.add_argument('--max_prop', type=int, default=2, help='Numero massimo di proprietà da analizzare')
    args = parser.parse_args()

    main(args.max_prop)
