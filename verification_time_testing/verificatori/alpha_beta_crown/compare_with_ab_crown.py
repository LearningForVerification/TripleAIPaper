import os
import csv
import logging
import argparse
import time
from jinja2 import Template
from complete_verifier.abcrown import ABCROWN
import signal
from contextlib import contextmanager
import io
from contextlib import redirect_stdout, redirect_stderr

# =========================
# Timeout per tipo di rete (in secondi)
# =========================
TIMEOUTS = {
    "CONV": 400,
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
# Logging (solo per debug, stampa principale sarà con print)
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# =========================
# Run ABCROWN
# =========================
# Variabile globale DEBUG
DEBUG = False  # True → stampa tutto l'output di ABCROWN, False → output minimal

def run_abcrown(model_path, property_path, template_path, timeout):
    """Esegue ABCROWN sul modello e sulla proprietà specificata con debug opzionale."""
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
            if DEBUG:
                # debug: lascia uscita standard
                ABCROWN(["--config", config_path]).main()
            else:
                # silenzioso: cattura output in memoria
                f_terminal = io.StringIO()
                with redirect_stdout(f_terminal), redirect_stderr(f_terminal):
                    ABCROWN(["--config", config_path]).main()

        elapsed = time.time() - start

    except TimeoutError:
        return timeout, "timeout"
    except Exception as e:
        if DEBUG:
            print(f"❌ Errore ABCROWN: {e}")
        return timeout, "error"

    # Controlla out.txt
    out_file = os.path.join(base_dir, "out.txt")
    if os.path.isfile(out_file):
        with open(out_file) as f_out:
            content = f_out.read().lower()
        os.remove(out_file)

        if "unsat" in content:
            status = "verified"
        elif "sat" in content:
            status = "not_verified"
        else:
            status = "unknown"
    else:
        status = "unknown"

    return elapsed, status


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
    DATASETS = ["FMNIST"]
    ARCHS = ["CONV", "2-FC"]
    SUBCATS = ["0.03", "over_param"]

    RESULTS_BASE = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS_BASE, exist_ok=True)

    TEMPLATE_PATH = os.path.join(BASE_DIR, "template_config_5.yaml")

    for dataset in DATASETS:
        property_folder = os.path.join(BASE_DIR, "properties", dataset, "0.03")
        if not os.path.isdir(property_folder):
            print(f"⚠️ Proprietà mancanti per {dataset}, skip")
            continue

        prop_files = sorted(f for f in os.listdir(property_folder) if f.endswith(".vnnlib"))[:args.max_prop]

        for arch in ARCHS:
            arch_path = os.path.join(BASE_DIR, "networks", dataset, arch)
            if not os.path.isdir(arch_path):
                continue

            arch_timeout = TIMEOUTS[arch]

            result_arch_path = os.path.join(RESULTS_BASE, arch, dataset)
            os.makedirs(result_arch_path, exist_ok=True)

            for subcat in SUBCATS:
                subcat_path = os.path.join(arch_path, subcat)
                if not os.path.isdir(subcat_path):
                    continue

                csv_path = os.path.join(result_arch_path, f"{subcat}.csv")

                with open(csv_path, "w", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(["model", "property", "status", "time"])

                    for nn_file in sorted(os.listdir(subcat_path)):
                        if not nn_file.endswith(".onnx"):
                            continue

                        nn_path = os.path.join(subcat_path, nn_file)
                        print(f"\n➡️ {dataset}/{arch}/{subcat} → {nn_file} (timeout={arch_timeout}s)")

                        times = []
                        num_timeout = 0
                        num_failure = 0

                        for prop in prop_files:
                            prop_path = os.path.join(property_folder, prop)

                            elapsed, status = run_abcrown(nn_path, prop_path, TEMPLATE_PATH, arch_timeout)

                            # Scrive su CSV
                            writer.writerow([nn_file, prop, status, elapsed])

                            # Stampa immediata a terminale in base allo stato
                            if status == "verified":
                                print(f"   ✅ {prop}: VERIFIED in {elapsed:.2f}s")
                                times.append(elapsed)
                            elif status == "not_verified":
                                print(f"   ❌ {prop}: NOT VERIFIED in {elapsed:.2f}s")
                                times.append(elapsed)
                            elif status == "timeout":
                                print(f"   ⏱ {prop}: TIMEOUT after {elapsed:.2f}s")
                                num_timeout += 1
                            elif status == "error":
                                print(f"   ⚠️ {prop}: ERROR")
                                num_failure += 1
                            else:
                                print(f"   ❓ {prop}: UNKNOWN in {elapsed:.2f}s")
                                num_failure += 1

                        # Riepilogo modello
                        median_time = sum(times) / len(times) if times else 0.0
                        writer.writerow([
                            nn_file,
                            "SUMMARY",
                            f"median_time={median_time:.2f}s, timeouts={num_timeout}, failures={num_failure}",
                            ""
                        ])
                        print(f"✅ {nn_file} | median={median_time:.2f}s | timeouts={num_timeout} | failures={num_failure}")

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    main()
