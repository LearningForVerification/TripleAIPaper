import subprocess


import sys
import os
sys.path.append(os.path.dirname(__file__))
from pynever.scripts.cli import sslp_verify_single, ssbp_verify_single

import time
import signal
from contextlib import contextmanager

from pynever.strategies.conversion.representation import load_network_path, ONNXNetwork
import os
import time
import signal
from contextlib import contextmanager
from pynever.strategies.conversion.converters.onnx import ONNXConverter
import pynever.networks as networks
import pynever.strategies.verification.algorithms as veralg
import pynever.strategies.verification.parameters as verparams
import pynever.strategies.verification.properties as verprop
from pynever.strategies.conversion.representation import load_network_path, ONNXNetwork


class TimeoutException(Exception):
    pass


@contextmanager
def with_timeout(seconds):
    def handler(signum, frame):
        raise TimeoutException(f"Timeout after {seconds} seconds")

    # Imposta il signal handler per SIGALRM
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)  # Imposta l'allarme

    try:
        yield
    finally:
        signal.alarm(0)  # Disabilita l'allarme


def ssbp_verify_single(model_file: str, property_file: str, timeout: int) -> tuple[bool, float]:
    nn_path = os.path.abspath(model_file)
    prop_path = os.path.abspath(property_file)

    if not os.path.isfile(nn_path):
        raise FileNotFoundError(f'Error: file {nn_path} not found!')
    if not os.path.isfile(prop_path):
        raise FileNotFoundError(f'Error: file {prop_path} not found!')

    alt_repr = load_network_path(nn_path)
    if not isinstance(alt_repr, ONNXNetwork):
        raise TypeError('The network is not an ONNX network!')

    network = ONNXConverter().to_neural_network(alt_repr)
    if not isinstance(network, networks.SequentialNetwork):
        raise TypeError('The network is not a sequential network!')

    prop = verprop.VnnLibProperty(prop_path)

    ver_params = verparams.SSBPVerificationParameters(timeout=timeout)
    ver_strategy = veralg.SSBPVerification(ver_params)

    start = time.perf_counter()
    result = ver_strategy.verify(network, prop)
    elapsed = time.perf_counter() - start

    # result.safe è un bool, ritorniamo tuple (bool, float)
    return result[0], elapsed


def get_pynever_time(model_path, property_path, timeout):
    nn_path = os.path.abspath(model_path)
    prop_path = os.path.abspath(property_path)

    if not os.path.isfile(nn_path):
        print(f'Error: file {nn_path} not found!')
        return "error", 0.0

    if not os.path.isfile(prop_path):
        print(f'Error: file {prop_path} not found!')
        return "error", 0.0

    try:
        start_time = time.time()
        #with with_timeout(timeout):
        is_verified, elapsed = ssbp_verify_single(model_path, property_path, timeout=timeout)
        elapsed = time.time() - start_time

        status = "verified" if is_verified else "not_verified"

    except TimeoutException:
        status = "out_of_time"
        elapsed = timeout

    except Exception as e:
        print(f"Unexpected error: {e}")
        status = "error"
        elapsed = time.time() - start_time

    if elapsed > timeout:
        status = "out_of_time"
        elapsed = timeout

    return status, elapsed


if __name__ == "__main__":
    model_path = "/path/to/your/model.onnx"
    property_path = "/path/to/your/property.vnnlib"
    timeout_seconds = 60

    status, elapsed = get_pynever_time(model_path, property_path, timeout_seconds)
    print(f"Status: {status}, Time elapsed: {elapsed:.2f} seconds")

if __name__ == "__main__":
    is_verified, elapsed_time = get_pynever_time(
        model_path=r"C:\Users\andr3\PycharmProjects\TripleAIPaper\verification_time_testing\networks\FC\0.03\flatten\30.onnx",
        property_path=r"C:\Users\andr3\PycharmProjects\TripleAIPaper\verification_time_testing\properties\0.03\sample_0011_label_6_eps_0.030.vnnlib",
        timeout=10,

    )

    print(f"Verified: {is_verified} | Time: {elapsed_time:.2f}s")
