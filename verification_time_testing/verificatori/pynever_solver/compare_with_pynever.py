import os
import csv
import logging
import argparse

from get_pynever_time import get_pynever_time


# =========================
# Main experiment
# =========================
def star_exp(max_prop, default_timeout):

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    logger = logging.getLogger()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATASETS = ["FMNIST", "CIFAR_CUSTOM"]
    ARCHS = ["FC", "2-FC"]
    SUBCATS = ["0.03", "over_param"]

    # -------------------------
    # Timeout per architettura
    # -------------------------
    TIMEOUTS = {
        "CONV": 180,
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

        prop_files = sorted(
            f for f in os.listdir(property_folder) if f.endswith(".vnnlib")
        )[:max_prop]

        for arch in ARCHS:
            arch_path = os.path.join(BASE_DIR, "networks", dataset, arch)
            if not os.path.isdir(arch_path):
                continue

            # Selezione timeout per architettura
            arch_timeout = TIMEOUTS.get(arch, default_timeout)

            logger.info(
                f"⏱️ Using timeout={arch_timeout}s for architecture {arch}"
            )

            result_arch_path = os.path.join(RESULTS_BASE, arch, dataset)
            os.makedirs(result_arch_path, exist_ok=True)

            for subcat in SUBCATS:
                subcat_path = os.path.join(arch_path, subcat)
                if not os.path.isdir(subcat_path):
                    continue

                csv_path = os.path.join(result_arch_path, f"{subcat}.csv")

                with open(csv_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["model", "property", "status", "time"])

                    for nn_file in sorted(os.listdir(subcat_path)):
                        if not nn_file.endswith(".onnx"):
                            continue

                        nn_path = os.path.join(subcat_path, nn_file)

                        logger.info(
                            f"➡️ {dataset}/{arch}/{subcat} → {nn_file}"
                        )

                        times = []
                        num_timeout = 0
                        num_failure = 0

                        for prop in prop_files:
                            prop_path = os.path.join(property_folder, prop)

                            status, elapsed = get_pynever_time(
                                nn_path,
                                prop_path,
                                timeout=arch_timeout
                            )

                            writer.writerow([nn_file, prop, status, elapsed])

                            if status in ("verified", "not_verified"):
                                times.append(elapsed)
                            elif status == "timeout":
                                num_timeout += 1
                            else:
                                num_failure += 1

                        median_time = (
                            sum(times) / len(times) if times else 0.0
                        )

                        writer.writerow([
                            nn_file,
                            "SUMMARY",
                            f"median_time={median_time:.2f}s, "
                            f"timeouts={num_timeout}, failures={num_failure}",
                            ""
                        ])

                        logger.info(
                            f"✅ {nn_file} | median={median_time:.2f}s | "
                            f"timeouts={num_timeout} | failures={num_failure}"
                        )


# =========================
# Entry point
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Verifica proprietà su reti neurali usando PyNEVer"
    )
    parser.add_argument(
        "--max_prop", type=int, default=1,
        help="Numero massimo di proprietà da verificare"
    )
    parser.add_argument(
        "--timeout", type=int, default=15,
        help="Timeout di default (fallback)"
    )
    args = parser.parse_args()

    star_exp(
        max_prop=args.max_prop,
        default_timeout=args.timeout
    )


if __name__ == "__main__":
    main()
