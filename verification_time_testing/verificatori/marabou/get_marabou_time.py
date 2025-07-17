
import logging
import signal
from datetime import time
import time


# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # log su console
    ]
)

import numpy as np
from maraboupy import Marabou



class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("Timeout scaduto!")

def get_marabou_time(onnx_path, vnnlib_path, timeout):
    """
    Verifica una proprietà VNN-LIB su una rete ONNX usando Marabou con un timeout forzato.
    """
    net = Marabou.read_onnx(onnx_path)

    options = Marabou.createOptions(
        verbosity=0,
        snc=True,
        numWorkers=1,
        initialTimeout=2,
        initialSplits=4,
        onlineSplits=8,
        timeoutInSeconds=timeout,
        timeoutFactor=1.5,
        splittingStrategy="auto",
        sncSplittingStrategy="largest-interval",
        tighteningStrategy="deeppoly",
        milpTightening="lp",
        dumpBounds=False,
        numSimulations=10,
        numBlasThreads=2,
        performLpTighteningAfterSplit=True,
        lpSolver="native"
    )

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        start = time.time()
        result = net.solve(propertyFilename=vnnlib_path, options=options)
        end = time.time()
        signal.alarm(0)  # Disabilita l'allarme

        elapsed = end - start
        status = result[0]

        if status == 'unsat':
            return elapsed, 'verified'
        elif status == 'sat':
            return elapsed, 'not_verified'
        else:
            return elapsed, status

    except TimeoutException:
        return timeout, 'timeout'

if __name__ == "__main__":
    import sys

    # Percorsi ai file modello e proprietà
    model_path = r"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/properties/networks/withotuh flatten/100.onnx"
    property_path = r"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/properties/properties/100/100_0.vnnlib"


    try:
        elapsed_time, status = get_marabou_time(model_path, property_path)

        print(f"\n--- RISULTATO VERIFICA ---")
        print(f"Modello   : {model_path}")
        print(f"Proprietà : {property_path}")
        print(f"Esito     : {status}")
        print(f"Tempo     : {elapsed_time:.2f} secondi\n")

    except Exception as e:
        print(f"Errore durante la verifica: {e}", file=sys.stderr)
