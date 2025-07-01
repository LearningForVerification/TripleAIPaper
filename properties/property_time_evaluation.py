import os
import subprocess
import time
import sys
import re
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor
import statistics

import pandas as pd
from jinja2 import Template
from pynever.scripts import cli
from pynever.networks import NeuralNetwork
import io
import contextlib
import concurrent.futures
import re
import os


def capture_output(func, *args, **kwargs):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        func(*args, **kwargs)
    return f.getvalue()


def parse_output(output: str):
    # Estrarre il tempo
    match_time = re.search(r"Verification Time:\s*([\d.]+)", output)
    verification_time = float(match_time.group(1)) if match_time else None

    # Estrarre il risultato finale
    match_result = re.search(r"Result:\s*(\w+)", output)
    result_str = match_result.group(1) if match_result else None

    if result_str.strip() == "Verified":
        is_sat = False
    else: is_sat = True

    verification_time = int(verification_time)

    return verification_time, is_sat

TIMEOUT = 300
MAX_PROPERTIES = 20
TEMPLATE_PATH = "../verification_time_testing/alpha_beta_crown/template_config.yaml"


def verify_property(model_path, property_file_path):
    # Load and render verification template
    with open(TEMPLATE_PATH) as f:
        template = Template(f.read())

    # Render template with model and property paths  
    config = template.render(
        onnx_path=model_path,
        vnnlib_path=property_file_path
    )

    # Save rendered config
    config_path = "config.yaml"
    with open(config_path, 'w') as f:
        f.write(config)

    cmd = [
        sys.executable,
        "complete_verifier/abcrown.py",
        "--config", config_path,
    ]
    try:
        start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        duration = time.time() - start_time
        
        # Read the out.txt file
        with open('out.txt', 'r') as f:
            content = f.read()
            
        # Check if 'sat' or 'unsat' is in the file
        if 'sat' in content and 'unsat' not in content:
            status = 'sat'
        elif 'unsat' in content:
            status = 'unsat'
        else:
            status = 'timeout'
            
    except subprocess.TimeoutExpired:
        return "timeout", TIMEOUT
    except FileNotFoundError:
        print("out.txt file not found")
        raise Exception("Error: out.txt file not found")

    if process.returncode != 0:
        raise Exception("Error running abcrown.py")

    return status, duration


def verify_properties(onnx_model_path, vnnlib_dir):
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    total_time_crown = 0.0
    total_time_never = 0.0
    results = []
    timeout_count = 0
    crown_timeout_count = 0
    never_timeout_count = 0

    model_name = os.path.basename(onnx_model_path)
    for i, vnnlib_file in enumerate(vnnlib_files[:MAX_PROPERTIES]):  # Global MAX_PROPERTIES = 10
        vnnlib_path = os.path.join(vnnlib_dir, vnnlib_file)
        print(f"Verifying property {i + 1}/{min(len(vnnlib_files), MAX_PROPERTIES)}: {vnnlib_file}")
        try:
            max_properties = MAX_PROPERTIES  # Set maximum properties to 10
            with ThreadPoolExecutor() as executor:
                # Crown verification with timeout
                crown_future = executor.submit(verify_property, onnx_model_path, vnnlib_path)
                try:
                    crown_status, crown_duration = crown_future.result(timeout=TIMEOUT)
                except concurrent.futures.TimeoutError:
                    crown_future.cancel()
                    crown_status, crown_duration = "TIMEOUT", TIMEOUT
                    timeout_count += 1
                    crown_timeout_count += 1

                # Never verification with timeout
                never_start_time = time.time()
                never_future = executor.submit(cli.verify_single_model, False, onnx_model_path, vnnlib_path, 'complete',
                                               'output.csv')
                try:
                    never_status = never_future.result(timeout=TIMEOUT)
                    never_duration = time.time() - never_start_time
                except concurrent.futures.TimeoutError:
                    never_future.cancel()
                    never_status, never_duration = "TIMEOUT", TIMEOUT
                    never_timeout_count += 1

            if crown_status != "TIMEOUT":
                total_time_crown += crown_duration
            if never_status != "TIMEOUT":
                total_time_never += never_duration

            print(f"Crown Result: {crown_status}, Time: {crown_duration:.2f} seconds")
            print(f"Never Result: {never_status}, Time: {never_duration:.2f} seconds")
            results.append((vnnlib_file, crown_status, crown_duration,
                            not never_status if never_status != "TIMEOUT" else "TIMEOUT", never_duration))
        except Exception as e:
            print(f"Error verifying {vnnlib_file}: {str(e)}")
            results.append((vnnlib_file, "ERROR", TIMEOUT, "ERROR", TIMEOUT))
            timeout_count += 1

    completed_count = len(vnnlib_files[:MAX_PROPERTIES]) - timeout_count
    avg_time_crown = total_time_crown / completed_count if completed_count else 0.0
    avg_time_never = total_time_never / completed_count if completed_count else 0.0

    print(f"\nProperties timed out: {timeout_count}")
    print(f"Average Crown verification time (excluding timeouts): {avg_time_crown:.2f} seconds")
    print(f"Average Never verification time (excluding timeouts): {avg_time_never:.2f} seconds")

    # Save results to CSV
    with open('verification_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Crown_Status', 'Crown_Duration', 'Never_Status', 'Never_Duration',
                         'Crown_Timeouts', 'Never_Timeouts'])
        for result in results:
            writer.writerow([model_name, result[1], f"{result[2]:.2f}", result[3], f"{result[4]:.2f}",
                             crown_timeout_count, never_timeout_count])

    return results

if __name__ == "__main__":

    # === Parametri iniziali ===
    model_hdim = 1000
    model_path = fr"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/properties/MNIST/MNIST-FC/models NOT overfitted/model_{model_hdim}.onnx"
    model_path = fr"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/properties/MNIST/MNIST-FC/best_models/{model_hdim}.onnx"

    vnnlib_dir = fr"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/properties/MNIST/PROPERTIES/properties/{model_hdim}"
    TIMEOUT = 300  # secondi

    # === Controlli iniziali ===
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(vnnlib_dir):
        print(f"VNNLib directory not found: {vnnlib_dir}")
        sys.exit(1)

    # === Carica il template YAML ===
    with open(TEMPLATE_PATH) as f:
        template = Template(f.read())

    # === Inizializzazione dati ===
    results = []
    times = []

    # === Loop su tutte le proprietà ===
    for filename in sorted(os.listdir(vnnlib_dir)):
        if not filename.endswith(".vnnlib"):
            continue

        vnnlib_path = os.path.join(vnnlib_dir, filename)
        config = template.render(onnx_path=model_path, vnnlib_path=vnnlib_path)

        with open("config.yaml", "w") as f:
            f.write(config)

        cmd = [
            sys.executable,
            "complete_verifier/abcrown.py",
            "--config", "config.yaml"
        ]

        try:
            start_time = time.time()
            subprocess.run(cmd, capture_output=False, text=True, timeout=TIMEOUT)
            duration = time.time() - start_time
        except subprocess.TimeoutExpired:
            results.append(("timeout", filename, TIMEOUT))
            times.append(TIMEOUT)
            continue

        # Leggi output
        with open("out.txt", "r") as f:
            content = f.read()

        if 'sat' in content and 'unsat' not in content:
            status = 'sat'
        elif 'unsat' in content:
            status = 'unsat'
        else:
            status = 'timeout'

        results.append((status, filename, duration))
        times.append(duration)

    # === Statistiche finali ===
    print("\n== Statistiche sui tempi ==")
    print(f"Numero proprietà verificate: {len(times)}")
    print(f"Tempo massimo: {max(times):.2f}s")
    print(f"Tempo minimo: {min(times):.2f}s")
    print(f"Tempo medio: {statistics.mean(times):.2f}s")
    print(f"Mediana: {statistics.median(times):.2f}s")
    if len(times) > 1:
        print(f"Deviazione standard: {statistics.stdev(times):.2f}s")

    # === Risultati per proprietà ===
    print("\n== Dettaglio per proprietà ==")
    for status, fname, t in results:
        print(f"{fname:40} -> {status.upper():7} | {t:.2f}s")

    # === Salvataggio su CSV ===
    df = pd.DataFrame(results, columns=["status", "filename", "duration_seconds"])
    df.to_csv("results.csv", index=False)
    print("\n📄 Risultati salvati in: results.csv")