import os
import time
import sys
from jinja2 import Template

# Assumendo che abcrown.py abbia una funzione principale che accetta un percorso di config
from complete_verifier.abcrown import ABCROWN

def get_alpha_beta_crown_time(model_path, property_path, template_path, timeout):
    # Carica template
    with open(template_path) as f:
        template = Template(f.read())

    config = template.render(
        onnx_path=model_path,
        vnnlib_path=property_path
    )

    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "config.yaml")

    # Scrivi config.yaml
    with open(config_path, 'w') as f:
        f.write(config)

    start_time = time.time()

    # Chiama direttamente la funzione principale di abcrown
    # Se abcrown_main accetta args simili a sys.argv, puoi passarli come lista
    ABCROWN(["--config", config_path]).main()
    elapsed = time.time() - start_time


    # Leggi risultato da out.txt
    out_file = os.path.join(current_dir, "out.txt")
    with open(out_file, "r") as f:
        content = f.read()

    if 'sat' in content and 'unsat' not in content:
        return elapsed, 'not_verified'
    elif 'unsat' in content:
        return elapsed, 'verified'
    else:
        return elapsed, 'failed'

def main():
    # Percorsi relativi
    model_rel = "networks/FC/0.03/100.onnx"
    property_rel = "properties/0.03/sample_0019_label_4_eps_0.030.vnnlib"
    template_rel = "template_config_3.yaml"
    timeout = 300  # Timeout in secondi

    # Converti in percorsi assoluti
    model_path = os.path.abspath(model_rel)
    property_path = os.path.abspath(property_rel)
    template_path = os.path.abspath(template_rel)

    print(f"Verifying model: {model_path}")
    print(f"Using property file: {property_path}")
    print(f"Using template: {template_path}")
    print(f"Timeout: {timeout} seconds")

    elapsed, status = get_alpha_beta_crown_time(model_path, property_path, template_path, timeout)

    print(f"Elapsed time: {elapsed} seconds")
    print(f"Verification status: {status}")

if __name__ == "__main__":
    main()