import os
import time
import multiprocessing

from pynever.strategies.verification.ssbp.constants import RefinementStrategy
from pynever.strategies.conversion.representation import load_network_path, ONNXNetwork
from pynever.strategies.conversion.converters.onnx import ONNXConverter
import pynever.networks as networks
import pynever.strategies.verification.algorithms as veralg
import pynever.strategies.verification.parameters as verparams
import pynever.strategies.verification.properties as verprop


def ssbp_verify_single_worker(model_file: str, property_file: str, timeout: int, return_dict):
    try:
        nn_path = os.path.abspath(model_file)
        prop_path = os.path.abspath(property_file)

        if not os.path.isfile(nn_path):
            raise FileNotFoundError(f'Model file not found: {nn_path}')
        if not os.path.isfile(prop_path):
            raise FileNotFoundError(f'Property file not found: {prop_path}')

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

        return_dict["result"] = result[0]
        return_dict["elapsed"] = elapsed

    except Exception as e:
        return_dict["result"] = None
        return_dict["elapsed"] = None
        return_dict["error"] = str(e)


def get_pynever_time(model_path, property_path, timeout):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    process = multiprocessing.Process(
        target=ssbp_verify_single_worker,
        args=(model_path, property_path, timeout, return_dict)
    )

    start = time.time()
    process.start()
    process.join(timeout)

    elapsed = time.time() - start

    if process.is_alive():
        process.terminate()
        process.join()
        status = "out_of_time"
        elapsed = timeout
    elif "error" in return_dict:
        print(f"Errore: {return_dict['error']}")
        status = "error"
    else:
        is_verified = return_dict.get("result")
        status = "verified" if is_verified else "not_verified"
        # elapsed preso da return_dict oppure calcolato
        elapsed = return_dict.get("elapsed", elapsed)

    print(f"Status: {status} | Time elapsed: {elapsed:.2f} seconds")
    return status, elapsed


if __name__ == "__main__":
    model_path = r"C:\Users\andr3\PycharmProjects\TripleAIPaper\verification_time_testing\networks\FC\0.03\flatten\30.onnx"
    property_path = r"C:\Users\andr3\PycharmProjects\TripleAIPaper\verification_time_testing\properties\0.03\sample_0011_label_6_eps_0.030.vnnlib"
    timeout_seconds = 10

    status, elapsed_time = get_pynever_time(model_path, property_path, timeout_seconds)
