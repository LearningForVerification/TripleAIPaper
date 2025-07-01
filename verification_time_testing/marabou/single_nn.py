import time
import os
import sys
from maraboupy import Marabou

TIMEOUT = 300  # secondi
OUTPUT_FILE = "verification_results.txt"

def verify_property(model_path, property_path):
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write("🧠 Caricamento modello...\n")
        net = Marabou.read_onnx(model_path)

        with open(OUTPUT_FILE, "a") as f:
            f.write("📜 Caricamento proprietà...\n")
            f.write(f"  ➤ Chiamata a solve su {property_path}\n")

        start_time = time.time()  # Inizio misurazione tempo
        result, _, stats = net.solve(filename='output', propertyFilename=property_path)
        end_time = time.time()  # Fine misurazione tempo
        elapsed_time = end_time - start_time  # Calcolo durata

        with open(OUTPUT_FILE, "a") as f:
            f.write(f"✅ Risultato: {result}\n")
            f.write(f"⏱ Tempo di esecuzione: {elapsed_time:.2f} secondi\n")

        if result == "unsat":
            status = "unsat"
        elif result == "sat":
            status = "sat"
        else:
            status = "unknown"

        return status, elapsed_time

    except TimeoutError:
        with open(OUTPUT_FILE, "a") as f:
            f.write("⏱ Timeout\n")
        return "timeout", None
    except Exception as e:
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"❌ Errore: {e}\n")
        return "error", -1


def main(model_path, property_path):
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"🔍 Verifica modello: {model_path}\n")
        f.write(f"  ➤ Proprietà: {property_path}\n")

    status, elapsed_time = verify_property(model_path, property_path)

    with open(OUTPUT_FILE, "a") as f:
        f.write(f"Status: {status}\n")
        f.write(f"Tempo totale solver: {elapsed_time:.2f} secondi\n")


if __name__ == "__main__":
    model_file = r'/mnt/c/Users/andr3/Desktop/primo-giugno-25/MNIST/MNIST-FC/temp/1000.onnx'
    property_file = "../generated_properties_MNIST/mnist_local_0.vnnlib"

    if not os.path.isfile(model_file):
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"Errore: modello non trovato '{model_file}'\n")
        sys.exit(1)

    if not os.path.isfile(property_file):
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"Errore: proprietà non trovata '{property_file}'\n")
        sys.exit(1)

    main(model_file, property_file)
