import argparse
import os
import pandas as pd
from verificatori.alpha_beta_crown.get_alpha_beta_crown_time import get_alpha_beta_crown_time
from verificatori.pynever.get_pynever_time import get_pynever_time
from verificatori.marabou.get_marabou_time import get_marabou_time
# from verificatori.pyrat.get_pyrat_time import get_pyrat_time

import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timecap(seconds):
    def handler(signum, frame):
        raise TimeoutException(f"⏰ Timeout after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def main(models_dir, properties_dir, timeout, output_dir):
    models = sorted(f for f in os.listdir(models_dir) if f.endswith(".onnx"))
    properties = sorted(f for f in os.listdir(properties_dir) if f.endswith(".vnnlib"))

    all_records = {
        "AlphaBetaCrown": [],
        # "PyRAT": [],
        "PyNeVer_ssbp": [],
        "PyNeVer_Overapproximated": [],
        "Marabou": [],
    }

    for model_file in models:
        model_path = os.path.join(models_dir, model_file)
        for prop_file in properties:
            prop_path = os.path.join(properties_dir, prop_file)
            print(f"\n▶️  Testing model '{model_file}' with property '{prop_file}'")

            # Alpha-beta Crown
            print("🔧 AlphaBetaCrown Solver")
            try:
                with timecap(timeout):
                    status, elapsed = get_alpha_beta_crown_time(model_path, prop_path, timeout)
            except TimeoutException:
                status, elapsed = "OUT_OF_TIME", timeout
            all_records["AlphaBetaCrown"].append({
                "model": model_file,
                "property": prop_file,
                "status": status,
                "time_seconds": elapsed
            })

            # PyNeVer (new heuristic)
            print("🔧 PyNeVer Solver New Heuristic (ssbp)")
            try:
                with timecap(timeout):
                    status, elapsed = get_pynever_time(model_path, prop_path, timeout, ver_abstraction_type="ssbp", strategy="")
            except TimeoutException:
                status, elapsed = "OUT_OF_TIME", timeout
            all_records["PyNeVer_ssbp"].append({
                "model": model_file,
                "property": prop_file,
                "status": status,
                "time_seconds": elapsed
            })

            # PyNeVer (overapproximated)
            print("🔧 PyNeVer Solver Overapproximated (sslp)")
            try:
                with timecap(timeout):
                    status, elapsed = get_pynever_time(model_path, prop_path, timeout, ver_abstraction_type="sslp", strategy="overapprox")
            except TimeoutException:
                status, elapsed = "OUT_OF_TIME", timeout
            all_records["PyNeVer_Overapproximated"].append({
                "model": model_file,
                "property": prop_file,
                "status": status,
                "time_seconds": elapsed
            })

            # Marabou
            print("🔧 Marabou Solver")
            try:
                with timecap(timeout):
                    status, elapsed = get_marabou_time(model_path, prop_path, timeout)
            except TimeoutException:
                status, elapsed = "OUT_OF_TIME", timeout
            all_records["Marabou"].append({
                "model": model_file,
                "property": prop_file,
                "status": status,
                "time_seconds": elapsed
            })

    # Salvataggio su file CSV separati
    os.makedirs(output_dir, exist_ok=True)
    for solver_name, records in all_records.items():
        df = pd.DataFrame(records)
        output_file = os.path.join(output_dir, f"{solver_name}_results.csv")
        df.to_csv(output_file, index=False)
        print(f"✅ Risultati '{solver_name}' salvati in '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch verification with multiple solvers")
    parser.add_argument("--models_dir", required=True, help="Directory with .onnx models")
    parser.add_argument("--properties_dir", required=True, help="Directory with .vnnlib properties")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per verifica (secondi)")
    parser.add_argument("--output_dir", default="verification_results", help="Directory per i file CSV di output")

    args = parser.parse_args()
    main(args.models_dir, args.properties_dir, args.timeout, args.output_dir)
