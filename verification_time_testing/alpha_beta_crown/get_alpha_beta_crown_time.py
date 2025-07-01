import argparse
import os
import sys
import time
import subprocess
import pandas as pd
from jinja2 import Template

TIMEOUT = 300  # secondi per ogni verifica

def verify_property(model_path, property_path, template_path):
    with open(template_path) as f:
        template = Template(f.read())

    config = template.render(
        onnx_path=model_path,
        vnnlib_path=property_path
    )

    with open("config.yaml", 'w') as f:
        f.write(config)

    cmd = [sys.executable, "complete_verifier/abcrown.py", "--config", "config.yaml"]

    start_time = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        duration = time.time() - start_time
    except subprocess.TimeoutExpired:
        return "timeout", TIMEOUT

    # Leggi risultato da out.txt
    with open("out.txt", "r") as f:
        content = f.read()

    if 'sat' in content and 'unsat' not in content:
        return 'sat', duration
    elif 'unsat' in content:
        return 'unsat', duration
    else:
        return 'unknown', duration

def main():


    model_folder = r'/mnt/c/Users/andr3/Desktop/primo-giugno-25/MNIST/MNIST-FC/sparse_models'
    property_folder = "../generated_properties_MNIST"
    template_path = "template_config.yaml"

    models = [f for f in os.listdir(model_folder) if f.endswith(".onnx")]
    properties = [f for f in os.listdir(property_folder) if f.endswith(".vnnlib")]

    all_results = []

    for model_file in models:
        model_path = os.path.join(model_folder, model_file)
        model_name = os.path.splitext(model_file)[0]
        print(f"\n🔍 Verificando modello: {model_name}")

        model_results = []
        for index, prop_file in enumerate(properties):
            if index == 30:
                break
            prop_path = os.path.join(property_folder, prop_file)
            print(f"  ➤ Proprietà: {prop_file} ...", end=' ', flush=True)

            try:
                status, duration = verify_property(model_path, prop_path, template_path)
                print(f"{status.upper()} ({duration:.2f}s)")
            except Exception as e:
                status = "error"
                duration = -1
                print(f"ERROR: {e}")

            model_results.append({
                "model": model_name,
                "property": prop_file,
                "status": status,
                "time_seconds": duration
            })

        # Calcolo del tempo massimo per questo modello
        max_time = max(r["time_seconds"] for r in model_results if r["time_seconds"] >= 0)
        for r in model_results:
            r["max_time_for_model"] = max_time

        all_results.extend(model_results)

    # Salva tutto in un unico CSV
    output_csv = os.path.join(model_folder, "all_verification_results.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\n📄 Tutti i risultati salvati in {output_csv}")

if __name__ == "__main__":
    main()
