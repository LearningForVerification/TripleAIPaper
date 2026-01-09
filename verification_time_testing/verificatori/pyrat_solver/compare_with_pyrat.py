import logging
import subprocess
import time
import argparse
import os
import csv
import re
import statistics
import multiprocessing as mp
import signal

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger()


# =========================
# Pyrat worker (isolato)
# =========================
def _pyrat_worker(model_path, property_path, timeout, queue):
    try:
        cmd = [
            "python", "pyrat.pyc",
            "--model_path", model_path,
            "--property_path", property_path,
            "--split", "True",
            "--verbose", "False",
            "--nb_process", "4",
            "--domains", "zonotopes",
            "--timeout", str(timeout)
        ]

        start = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        elapsed = time.time() - start

        match = re.search(r"Result\s*=\s*(True|False)", result.stdout)
        print(f"pyrat_response: {result.stdout}")
        if match:
            status = "verified" if match.group(1) == "True" else "not_verified"
        else:
            status = "error"

        queue.put((elapsed, status))

    except Exception as e:
        queue.put((0.0, f"error:{e}"))


# =========================
# Pyrat wrapper con timeout HARD
# =========================
import subprocess
import time
import re
import logging

logger = logging.getLogger()

def run_pyrat(model_path, property_path, timeout):
    cmd = [
        "python", "pyrat.pyc",
        "--model_path", model_path,
        "--property_path", property_path,
        "--split", "True",
        "--verbose", "False",
        "--nb_process", "4",
        "--domains", "zonotopes"
    ]

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout

        )

        elapsed = time.time() - start
        output = result.stdout + result.stderr
        print(f"{output}")
        match = re.search(r"Result\s*=\s*(True|False)", output)


        if match:
            status = "verified" if match.group(1) == "True" else "not_verified"
        else:
            status = "error"

        return elapsed, status

    except subprocess.TimeoutExpired:
        logger.warning("⏱️ Timeout → Pyrat terminato")
        return timeout, "timeout"

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

    DATASETS = ["CIFAR_CUSTOM", "FMNIST"]
    ARCHS = ["FC", "2-FC", "CONV"]
    SUBCATS = ["0.03", "over_param"]

    # Timeout per categoria
    TIMEOUTS = {
        "CONV": 180,
        "FC": 15,
        "2-FC": 180
    }


    RESULTS_BASE = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS_BASE, exist_ok=True)

    for dataset in DATASETS:

        # Proprietà (come Marabou)
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

            # results/<ARCH>/<DATASET>/
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
