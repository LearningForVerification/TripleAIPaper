import argparse
import os
import sys
import time
import subprocess
import pandas as pd
from jinja2 import Template

TIMEOUT = 15  # secondi per ogni verifica

def get_alpha_beta_crown_time(model_path, property_path, template_path, timeout):
    with open(template_path) as f:
        template = Template(f.read())

    config = template.render(
        onnx_path=model_path,
        vnnlib_path=property_path
    )

    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "config.yaml")

    with open(config_path, 'w') as f:
        f.write(config)

    cmd = [sys.executable, "complete_verifier/abcrown.py", "--config", config_path]

    start_time = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        elapsed = time.time() - start_time
    except subprocess.TimeoutExpired:
        return "timeout", TIMEOUT

    # Leggi risultato da out.txt
    with open(os.path.join(os.path.dirname(__file__), "out.txt"), "r") as f:
        content = f.read()

    if 'sat' in content and 'unsat' not in content:
        return elapsed, 'not_verified'
    elif 'unsat' in content:
        return elapsed, 'verified'
    else:
        return elapsed, 'failed'

