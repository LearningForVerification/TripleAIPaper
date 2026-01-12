import logging
import subprocess
import time
import argparse
import os
import csv
import re
import statistics
import multiprocessing as mp

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger()


# =========================
# Pyrat runner con output live e gestione timeout
# =========================
def run_pyrat(model_path, property_path, timeout, nb_process=4, domains="zonotopes", split=True, verbose=False):
    # Aggiungiamo --timeout al comando
    cmd = [
        "python", "pyrat.pyc",
        "--model_path", model_path,
        "--property_path", property_path,
        "--split", str(split),
        "--verbose", str(verbose),
        "--nb_process", str(nb_process),
        "--domains", str(domains),
        "--timeout", str(timeout)   # <-- timeout passato direttamente a Pyrat
    ]

    logger.info(f"➡️ Comando: {' '.join(cmd)}")
    start = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output = ""
        try:
            # stampiamo tutto in tempo reale
            for line in process.stdout:
                print(line, end="")  # output live
                output += line

            # attendiamo fine processo con timeout HARD
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            logger.warning(f"⏱️ Timeout HARD: {model_path} terminato dopo {timeout}s")
            return timeout, "timeout"

        elapsed = time.time() - start

        # regex per catturare anche Timeout interno di Pyrat
        match = re.search(r"Result\s*=\s*(True|False|Timeout)", output, re.IGNORECASE)
        if match:
            val = match.group(1).lower()
            if val == "true":
                status = "verified"
            elif val == "false":
                status = "not_verified"
            elif val == "timeout":
                status = "timeout"
        else:
            status = "error"

        return elapsed, status

    except Exception as e:
        logger.error(f"❌ Errore Pyrat: {e}")
        return 0.0, "error"


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_prop", type=int, default=1)
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATASETS = ["FMNIST"]
    ARCHS = ["2-FC", "CONV"]
    SUBCATS = ["0.03", "over_param"]

    # Timeout personalizzati per architettura
    TIMEOUTS = {
        "CONV": 400,
        "FC": 15,
        "2-FC": 180
    }

    RESULTS_BASE = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS_BASE, exist_ok=True)

    for dataset in DATASETS:
        property_folder = os.path.join(BASE_DIR, "properties", dataset, "0.03")
        if not os.path.isdir(property_folder):
            logger.warning(f"⚠️ Proprietà mancanti per {dataset}, skip")
            continue

        prop_files = sorted(os.listdir(property_folder))[:args.max_prop]

        for arch in ARCHS:
            arch_path = os.path.join(BASE_DIR, "networks", dataset, arch)
            if not os.path.isdir(arch_path):
                continue

            timeout_for_arch = TIMEOUTS.get(arch, 50)

            result_arch_path = os.path.join(RESULTS_BASE, arch, dataset)
            os.makedirs(result_arch_path, exist_ok=True)

            for subcat in SUBCATS:
                subcat_path = os.path.join(arch_path, subcat)
                if not os.path.isdir(subcat_path):
                    continue

                csv_path = os.path.join(result_arch_path, f"{subcat}.csv")

                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["model", "property", "status", "time"])

                    for nn_file in sorted(os.listdir(subcat_path)):
                        if not nn_file.endswith(".onnx"):
                            continue

                        nn_path = os.path.join(subcat_path, nn_file)

                        logger.info(
                            f"➡️ {dataset}/{arch}/{subcat} → {nn_file} | timeout={timeout_for_arch}s"
                        )

                        times = []
                        num_timeout = 0
                        num_failure = 0

                        for prop in prop_files:
                            prop_path = os.path.join(property_folder, prop)

                            elapsed, status = run_pyrat(
                                nn_path, prop_path, timeout_for_arch
                            )

                            writer.writerow([nn_file, prop, status, elapsed])

                            if status in ("verified", "not_verified"):
                                times.append(elapsed)
                            elif status == "timeout":
                                num_timeout += 1
                            else:
                                num_failure += 1

                        median_time = statistics.median(times) if times else 0.0

                        # Summary stile Marabou
                        writer.writerow([
                            nn_file,
                            "SUMMARY",
                            f"median_time={median_time:.2f}s, timeouts={num_timeout}, failures={num_failure}",
                            ""
                        ])

                        logger.info(
                            f"✅ {nn_file} | median={median_time:.2f}s | "
                            f"timeouts={num_timeout} | failures={num_failure}"
                        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
