import argparse
import os
import sys
import time
import subprocess
from jinja2 import Template

# Timeout di default (secondi) per ogni verifica
TIMEOUT = 15

def get_alpha_beta_crown_time(model_path, property_path, template_path, timeout=TIMEOUT):
    """
    Esegue ABCROWN sul modello ONNX e sulla proprietà VNNLIB.

    Parametri:
    -----------
    model_path : str
        Percorso del file ONNX del modello (assoluto o relativo a questo script).
    property_path : str
        Percorso del file VNNLIB della proprietà (assoluto o relativo a questo script).
    template_path : str
        Percorso del template YAML di configurazione ABCROWN (assoluto o relativo a questo script).
    timeout : float
        Tempo massimo (in secondi) per la verifica.

    Ritorna:
    --------
    elapsed : float o int
        Tempo impiegato per la verifica.
    status : str
        'verified', 'not_verified', 'failed' o 'timeout'.
    """

    # Risolvo percorsi assoluti relativi a questo script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, model_path) if not os.path.isabs(model_path) else model_path
    property_path = os.path.join(base_dir, property_path) if not os.path.isabs(property_path) else property_path
    template_path = os.path.join(base_dir, template_path) if not os.path.isabs(template_path) else template_path

    # 1️⃣ Leggi e renderizza il template di configurazione
    with open(template_path, "r") as f:
        template = Template(f.read())
    config = template.render(onnx_path=model_path, vnnlib_path=property_path)

    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(config)

    # 2️⃣ Comando per eseguire ABCROWN
    abcrown_path = os.path.join(base_dir, "complete_verifier", "abcrown.py")
    cmd = [sys.executable, abcrown_path, "--config", config_path]

    start_time = time.time()
    try:
        # Esegui ABCROWN catturando stdout/stderr
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start_time

        # Stampa stdout/stderr per debug
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    except subprocess.TimeoutExpired:
        return timeout, "timeout"
    except Exception as e:
        print(f"[ERROR] Exception durante ABCROWN: {e}")
        return timeout, "failed"

    # 3️⃣ Controllo out.txt (alcune versioni di ABCROWN scrivono qui)
    out_file = os.path.join(base_dir, "out.txt")
    content = ""
    if os.path.isfile(out_file):
        with open(out_file, "r") as f:
            content = f.read().lower()

    # 4️⃣ Determina lo status
    combined_content = content + "\n" + result.stdout.lower()
    if "unsat" in combined_content:
        return elapsed, "verified"
    elif "sat" in combined_content:
        return elapsed, "not_verified"
    elif "timeout" in combined_content:
        return elapsed, "timeout"
    else:
        return elapsed, "failed"


# =========================
# Esempio di utilizzo
# =========================
if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model = os.path.join(base_dir, "networks/FMNIST/2-FC/0.03/2x50.onnx")
    prop = os.path.join(base_dir, "properties/FMNIST/0.03/sample_0000_label_9_eps_0.0300.vnnlib")
    template = os.path.join(base_dir, "template_config_5.yaml")
    timeout = 15

    elapsed, status = get_alpha_beta_crown_time(model, prop, template, timeout)
    print(f"Risultato: {status} | Tempo: {elapsed:.2f}s")
