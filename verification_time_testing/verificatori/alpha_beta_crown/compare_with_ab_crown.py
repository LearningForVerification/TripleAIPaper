import os
import csv
import logging
import argparse
import time
from jinja2 import Template
from complete_verifier.abcrown import ABCROWN
import signal
from contextlib import contextmanager

# =========================
# Timeout per tipo di rete (in secondi)
# =========================
TIMEOUTS = {
    "CONV": 180,
    "FC": 15,
    "2-FC": 180
}

# =========================
# Timeout context (Unix only)
# =========================
@contextmanager
def timer(seconds: float):
    """Raise TimeoutError if block takes longer than seconds."""
    def _handler(signum, frame):
        raise TimeoutError(f"Timeout after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger()

# =========================
# Run ABCROWN
# =========================
def run_abcrown(model_path, property_path, template_path, timeout):
    """Esegue ABCROWN sul modello e sulla proprietà specificata."""
    with open(template_path) as f:
        template = Template(f.read())

    config = template.render(
        onnx_path=model_path,
        vnnlib_path=property_path
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")

    with open(config_path, "w") as f:
        f.write(config)

    start = time.time()
    try:
        with timer(timeout):
            ABCROWN(["--config", config_path]).main()
        elapsed = time.time() - start

    except TimeoutError:
        return timeout, "timeout"
    except Exception:
        return timeout, "error"

    out_file = os.path.join(base_dir, "out.txt")
    if not os.path.isfile(out_file):
        return elapsed, "failed"

    with open(out_file) as f:
        content = f.read().lower()

    if "unsat" in content:
        return elapsed, "verified"
    if "sat" in content:
        return elapsed, "not_verified"
    if "timeout" in content:
        return elapsed, "unknown"

    return elapsed, "failed"

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Verifica proprietà su reti neurali usando ABCROWN")
    parser.add_argument("--max_prop", type=int, default=100, help="Numero massimo di proprietà da verificare")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATASETS = ["FMNIST", "CIFAR_CUSTOM"]
    ARCHS = ["CONV", "FC", "2-FC"]
    SUBCATS = ["0.03", "over_param"]

    RESULTS_BASE = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS_BASE, exist_ok=True)

    TEMPLATE_PATH = os.path.join(BASE_DIR, "template_config_5.yaml")

    for dataset in DATASETS:
        property_folder = os.path.join(BASE_DIR, "properties", dataset, "0.03")
        if not os.path.isdir(property_folder):
            logger.warning(f"⚠️ Proprietà mancanti per {dataset}, skip")
            continue

        prop_files = sorted(f for f in os.listdir(property_folder) if f.endswith(".vnnlib"))[:args.max_prop]

        for arch in ARCHS:
            arch_path = os.path.join(BASE_DIR, "networks", dataset, arch)
            if not os.path.isdir(arch_path):
                continue

            # Timeout fisso per architettura dal dizionario
            arch_timeout = TIMEOUTS[arch]

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
                        logger.info(f"➡️ {dataset}/{arch}/{subcat} → {nn_file} (timeout={arch_timeout}s)")

                        times = []
                        num_timeout = 0
                        num_failure = 0

                        for prop in prop_files:
                            prop_path = os.path.join(property_folder, prop)
                            elapsed, status = run_abcrown(nn_path, prop_path, TEMPLATE_PATH, arch_timeout)

                            writer.writerow([nn_file, prop, status, elapsed])

                            if status in ("verified", "not_verified"):
                                times.append(elapsed)
                            elif status == "timeout":
                                num_timeout += 1
                            else:
                                num_failure += 1

                        median_time = sum(times) / len(times) if times else 0.0

                        writer.writerow([
                            nn_file,
                            "SUMMARY",
                            f"median_time={median_time:.2f}s, timeouts={num_timeout}, failures={num_failure}",
                            ""
                        ])

                        logger.info(f"✅ {nn_file} | median={median_time:.2f}s | timeouts={num_timeout} | failures={num_failure}")

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    main()
