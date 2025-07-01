import os
import time
from multiprocessing import Process, Queue

import pandas as pd
import logging
from maraboupy import Marabou

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # log su console
    ]
)

TIMEOUT = 300  # secondi

# === Configurazione ===
MODEL_FOLDER = r'/mnt/c/Users/andr3/Desktop/primo-giugno-25/MNIST/MNIST-FC/sparse_models'
PROPERTY_FOLDER = "../generated_properties_MNIST"
OUTPUT_CSV = os.path.abspath("summary_max_times.csv")

# === Logging ===
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# === Funzione per chiamare solve() in processo separato ===
def run_verification(model_path, property_path, queue):
    try:
        net = Marabou.read_onnx(model_path)
        start_time = time.time()
        result, _, stats = net.solve(filename='output', propertyFilename=property_path)
        duration = time.time() - start_time

        if result == "unsat":
            queue.put(("unsat", duration))
        elif result == "sat":
            queue.put(("sat", duration))
        else:
            queue.put(("unknown", duration))
    except Exception as e:
        queue.put(("error", -1))

from queue import Empty

# === Funzione che gestisce processo e timeout ===
def verify_property(model_path, property_path, timeout=TIMEOUT):
    queue = Queue()
    p = Process(target=run_verification, args=(model_path, property_path, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return "timeout", timeout
    else:
        try:
            result = queue.get_nowait()
            return result
        except Empty:
            return "error", "No result returned from process"


# === Main ===
def main():
    models = sorted([f for f in os.listdir(MODEL_FOLDER) if f.endswith(".onnx")])
    properties = sorted([f for f in os.listdir(PROPERTY_FOLDER) if f.endswith(".vnnlib")])

    summary_results = []

    for model_file in models:
        model_path = os.path.join(MODEL_FOLDER, model_file)
        model_name = os.path.splitext(model_file)[0]
        logging.info(f"🔍 Verificando modello: {model_name}")

        max_time = -1

        for index, prop_file in enumerate(properties):
            if index == 30:
                break  # solo prime 30 proprietà
            prop_path = os.path.join(PROPERTY_FOLDER, prop_file)
            logging.info(f"  ➤ Proprietà: {prop_file}")

            try:
                status, duration = verify_property(model_path, prop_path)
                logging.info(f"     ➝ {status.upper()} ({duration:.2f}s)")
                if duration > max_time:
                    max_time = duration
            except Exception as e:
                logging.error(f"     ✖ Errore nella verifica: {e}")
                continue

        summary_results.append({
            "model": model_name,
            "max_time_seconds": round(max_time, 2)
        })

    df = pd.DataFrame(summary_results)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"\n📄 Risultati salvati in: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
