import argparse
import csv
import logging
import os
import signal
from datetime import time
import time
import numpy as np
from maraboupy import Marabou



# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # log su console
    ]
)
logger = logging.getLogger()

import time
import multiprocessing as mp
from maraboupy import Marabou


def run_solver(onnx_path, vnnlib_path, options, queue):
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
    """
    Verifica una proprietà VNN-LIB su una rete ONNX usando Marabou con un timeout robusto.
    """

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
        p.terminate()  # prova prima con terminate (SIGTERM)
        time.sleep(0.5)
        print("Trying to kill the process...")
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)  # forzato kill (SIGKILL)
        p.join()
        return timeout, 'timeout'

    if queue.empty():
        return timeout, 'timeout'

    return queue.get()


def main(max_prop, timeout):
    current_directory = os.path.dirname(os.path.abspath(__file__))

    experiments_category_folders = ["2-FC", "FC"]
    experiments_category_folders = [os.path.join(current_directory, "networks", x) for x in experiments_category_folders]

    sub_category_folder = ["0.03", "not_over_param", "over_param", "not_over_param_not_sparse"]

    property_folder = os.path.join(current_directory, "properties", "0.03")
    if not os.path.isdir(property_folder):
        raise Exception(f"Directory '{property_folder}' not found")

    for folder in experiments_category_folders:
        if not os.path.isdir(folder):
            raise Exception(f"Directory '{folder}' not found")

        for sub_folder in sub_category_folder:
            sub_path = os.path.join(folder, sub_folder)
            if not os.path.isdir(sub_path):
                raise Exception(f"Directory '{sub_path}' not found")

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

                        template_path = os.path.join(current_directory, "template_config.yaml")
                        elapsed, status = get_marabou_time(nn_path, prop_path, timeout=timeout)

                        writer.writerow([nn_file, prop_file, status, elapsed])

                    logger.info(f"✅ Completata rete: {nn_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valutazione reti con alpha_beta_crown.")
    parser.add_argument('--max_prop', type=int, default=300, help='Numero massimo di proprietà da analizzare')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout per l\'analisi di ogni proprietà (in secondi)')
    args = parser.parse_args()

    main(args.max_prop, args.timeout)

