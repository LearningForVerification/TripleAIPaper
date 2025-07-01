import os
import time
import pandas as pd
import concurrent.futures
from pynever.scripts import cli

# === Parametri ===
TIMEOUT = 300  # secondi
MAX_PROPERTIES = 1  # opzionale, None per tutti
MODALITY = "mixed"

def verify_with_pynever(model_path, property_path):
    start_time = time.time()
    cli.sslp_verify_single(
        safety_prop=False,
        model_file=model_path,
        property_file=property_path,
        strategy=MODALITY,
        logfile="log.csv"
    )
    duration = time.time() - start_time



def verify_model_with_all_properties(model_path, property_dir):
    vnnlib_files = sorted([f for f in os.listdir(property_dir) if f.endswith('.vnnlib')])
    if MAX_PROPERTIES:
        vnnlib_files = vnnlib_files[:MAX_PROPERTIES]

    max_duration = 0
    statuses = []

    for i, filename in enumerate(vnnlib_files):
        prop_path = os.path.join(property_dir, filename)
        print(f"🔍 Verifying {os.path.basename(model_path)} with {filename}... ", end="")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(verify_with_pynever, model_path, prop_path)
            try:
                status, duration = future.result(timeout=TIMEOUT)
            except concurrent.futures.TimeoutError:
                status, duration = "timeout", TIMEOUT

        print(f"{status.upper()} ({duration:.2f}s)")
        max_duration = max(max_duration, duration)
        statuses.append(status)

    return {
        "model": os.path.basename(model_path),
        "max_duration_seconds": max_duration,
        "all_verified": all(s == "verified" for s in statuses),
        "some_failed": any(s != "verified" for s in statuses),
        "num_properties": len(vnnlib_files)
    }

def verify_all_models(model_dir, property_dir):
    onnx_models = sorted([f for f in os.listdir(model_dir) if f.endswith('.onnx')])
    results = []

    for model_file in onnx_models:
        model_path = os.path.join(model_dir, model_file)
        result = verify_model_with_all_properties(model_path, property_dir)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("pynever_all_models_results.csv", index=False)
    print("\n📄 Risultati salvati in: pynever_all_models_results.csv")

if __name__ == "__main__":
    # 📁 Imposta qui i percorsi
    model_dir = r'/mnt/c/Users/andr3/Desktop/primo-giugno-25/MNIST/MNIST-FC/best_models'
    model_dir = r"C:\Users\andr3\Desktop\primo-giugno-25\MNIST\MNIST-FC\best_models\flatten"
    property_dir = "../generated_properties_MNIST"

    if not os.path.exists(model_dir):
        print(f"Cartella modelli non trovata: {model_dir}")
        exit(1)

    if not os.path.exists(property_dir):
        print(f"Cartella proprietà non trovata: {property_dir}")
        exit(1)

    verify_all_models(model_dir, property_dir)
