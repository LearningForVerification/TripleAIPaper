import contextlib
import os
import random
import warnings

import numpy as np
import torch
# from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
# from auto_LiRPA.utils import MultiAverageMeter
# from exceptiongroup import suppress
from torch import optim, nn, multiprocessing
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from utils.dataset import get_dataset
from utils.utils import load_yaml_config, write_results_on_csv

DATASET_DIRECTORY = '../../datasets'
OUTPUT_FOLDER = r"networks"

# Setting seed
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

# Set the device (use GPU if available, otherwise fallback to CPU)
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

# Set of Control Variaibles
REFINEMENT_THRESHOLD = 9
REFINEMENT_FACTOR = 10
LAMBDA_ITERATION_LIMIT = 2
NOISE_ITERATION_LIMIT = 2
LAMBDA_PERCENTAGE_INCREASING = 2/100
NOISE_PERCENTAGE_INCREASING = 2/100
N_EPOCHS = (REFINEMENT_THRESHOLD + 1) + LAMBDA_ITERATION_LIMIT * REFINEMENT_FACTOR + NOISE_ITERATION_LIMIT * LAMBDA_ITERATION_LIMIT * REFINEMENT_FACTOR



# Print all variable values
print("REFINEMENT_THRESHOLD =", REFINEMENT_THRESHOLD)
print("REFINEMENT_FACTOR =", REFINEMENT_FACTOR)
print("LAMBDA_ITERATION_LIMIT =", LAMBDA_ITERATION_LIMIT)
print("NOISE_ITERATION_LIMIT =", NOISE_ITERATION_LIMIT)
print("LAMBDA_PERCENTAGE_INCREASING =", LAMBDA_PERCENTAGE_INCREASING)
print("NOISE_PERCENTAGE_INCREASING =", NOISE_PERCENTAGE_INCREASING)
print("N_EPOCHS =", N_EPOCHS)




def calculate_epochs(filters_number):
    return N_EPOCHS


def get_min_index_and_value(list_of_best_results):
    best_index, best_tuple = min(
        enumerate(list_of_best_results),
        key=lambda x: x[1][1]['n_unstable_nodes']
    )
    return list_of_best_results[best_index][0], list_of_best_results[best_index][1]


def generate_networks(data_dict, dataset_name, hidden_dim, rs_factor, calculate_epochs, to_beat_metrics=None):

    # Unpack data_dict
    optimizer_dict = data_dict['optimizer']
    scheduler_lr_dict = data_dict['scheduler_lr']

    # Create optimizer params dict
    opt_params = optimizer_dict.copy()
    optimizer_name = opt_params['type']
    del opt_params['type']

    opt_params = opt_params.copy()

    # Create scheduler_lr_dict params dict
    scheduler_lr_params = scheduler_lr_dict.copy()
    scheduler_lr_name = scheduler_lr_params['type']
    del scheduler_lr_params['type']

    # Dataset parameters
    train_dim = int(data_dict['data']['train_dim'])
    test_dim = int(data_dict['data']['test_dim'])

    # NN architectures
    input_dim = int(data_dict['data']['input_dim'])
    output_dim = int(data_dict['data']['output_dim'])

    # Training parameters
    train_batch_size = int(data_dict['training']['train_batch_size'])
    test_batch_size = int(data_dict['training']['test_batch_size'])

    loss_name = data_dict['training']['loss_name']

    train_set, test_det, dummy_input = get_dataset(dataset_name)


    # Create a subset
    train_dataset = Subset(train_set, range(train_dim))
    test_subset = Subset(test_set, range(test_dim))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    test_data_loader = torch.utils.data.DataLoader(test_subset, batch_size=test_batch_size, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))

    # Create model_ori
    model_ori = SimpleFCNet(hidden_dim).to(device)

    # BoundedModule senza ottimizzazione
    lirpa_model = BoundedModule(model_ori, dummy_input, bound_opts={"enable_alpha_crown": True, "relu": "adaptive"})

    if optimizer_name == 'Adam':
        optimizer_cls = optim.Adam
    elif optimizer_name == 'SGD':
        optimizer_cls = optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Define the loss function
    if loss_name == 'CrossEntropyLoss':
        criterion_cls = nn.CrossEntropyLoss
    elif loss_name == 'MSE':
        criterion_cls = nn.MSELoss
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    criterion_callable = criterion_cls()

    metrics = train(model_lirpa=lirpa_model, device=device, train_loader=train_data_loader, test_loader=test_data_loader,
                    optimizer_cls=optimizer_cls, optimizer_params=opt_params, criterion_cls=criterion_cls,
                    num_epochs=calculate_epochs(hidden_dim), hidden_layer_dim=hidden_dim, num_classes=10, rs_loss_factor=rs_factor, to_beat_metric=to_beat_metrics)

    return metrics, lirpa_model

def compute_rs_loss(lb, ub, hidden_layer_dim, norm_constant=1.0, normalized=True):
    rs_loss = -torch.mean(torch.sum(torch.tanh(1.0 + norm_constant * lb * ub), dim=-1))

    if rs_loss < lb.shape[1] * -1 or rs_loss > lb.shape[1]:
        raise Exception("Error in RS Loss, value exceeding the maximum")

    n_unstable_nodes = (lb * ub < 0).sum(dim=1).float().mean().item()
    #print(f"{n_unstable_nodes=}")

    if normalized:
        rs_loss = rs_loss / hidden_layer_dim
        rs_loss = (rs_loss + 1) / 2
        assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"

    return rs_loss, n_unstable_nodes


def create_checkpoint(model, optimizer, epoch, scheduler=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    return checkpoint


def load_checkpoint(model, optimizer, checkpoint, device, scheduler=None):
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


def train(model_lirpa, device, train_loader, test_loader, optimizer_cls, optimizer_params, criterion_cls, num_epochs,
          hidden_layer_dim, num_classes, rs_loss_factor=None, eps=0.015, to_beat_metric = None):
    optimizer = optimizer_cls(model_lirpa.parameters(), **optimizer_params)
    criterion = criterion_cls()
    refinement_bool = False
    eps = eps
    test_eps = eps
    rs_loss_factor = rs_loss_factor

    metrics = {
        'train_loss': None, 'train_accuracy': None, 'test_loss': None, 'test_accuracy': None,
        'loss_on_train': None, 'rs_loss_on_train': None, 'loss_on_test': None, 'rs_loss_on_test': None,
        'lambda': [], 'eps': [], 'number_of_rollback': 0
    }

    metrics['lambda'].append(rs_loss_factor)
    metrics['eps'].append(eps)

    if to_beat_metric is not None:
        refinement_bool = True
        previous_acc = to_beat_metric["test_accuracy"]
        previous_unstable_nodes = to_beat_metric["n_unstable_nodes"]
        checkpoint = None
        lambda_counter = 0
        noise_counter = 0
        
    i_epoch = 0
    while i_epoch < num_epochs:

        model_lirpa.train()
        running_train_loss, running_train_loss_1, running_train_loss_2 = 0.0, 0.0, 0.0
        correct_train, total_train = 0, 0

        # Noise type
        perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)
        test_perturbation = PerturbationLpNorm(norm=np.inf, eps=test_eps)

        for index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_lirpa(inputs)

            targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
            loss = criterion(outputs, targets_hot_encoded) if isinstance(criterion, nn.MSELoss) else criterion(outputs,
                                                                                                               targets)


            partial_loss_1, partial_loss_2 = loss.item(), 0.0
            if rs_loss_factor is not None:

                # Input con perturbazione
                x_perturbed = BoundedTensor(inputs, perturbation)

                # Calcolo bound
                _, _ = model_lirpa.compute_bounds(x=(x_perturbed,),
                                                    method='IBP')
                save_dict = model_lirpa.save_intermediate()

                lb, ub = save_dict.get('/6')

                train_rs_loss, _ = compute_rs_loss(lb, ub, hidden_layer_dim)

                loss += rs_loss_factor * train_rs_loss
                #print(f"{rs_loss_factor=}")
                partial_loss_2 = train_rs_loss.item()
                #print(f"{train_rs_loss=}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_loss_1 += partial_loss_1
            running_train_loss_2 += partial_loss_2

            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        loss_1_train = running_train_loss_1 / len(train_loader)
        loss_2_train = running_train_loss_2 / len(train_loader)

        metrics.update({
            'train_loss': train_loss, 'train_accuracy': train_accuracy,
            'loss_on_train': loss_1_train, 'rs_loss_on_train': loss_2_train
        })

        print(f"Epoch {i_epoch + 1}:")
        print(f"  Train -> Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        if i_epoch % REFINEMENT_FACTOR//2 == 0 or i_epoch == num_epochs - 1:
            model_lirpa.eval()
            running_test_loss, running_test_loss_1, running_test_loss_2 = 0.0, 0.0, 0.0
            correct_test, total_test = 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model_lirpa(inputs)

                    targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                    loss = criterion(outputs, targets_hot_encoded) if isinstance(criterion, nn.MSELoss) else criterion(
                        outputs, targets)

                    partial_loss_1, partial_loss_2 = loss.item(), 0.0


                    if i_epoch == num_epochs - 1 or i_epoch % REFINEMENT_FACTOR  == 0:
                        if rs_loss_factor is not None:
                            # Input con perturbazione
                            x_perturbed = BoundedTensor(inputs, test_perturbation)

                            # Calcolo bound
                            # Sopprimere il warning solo durante la chiamata
                            with open(os.devnull, 'w') as fnull:
                                with contextlib.redirect_stdout(fnull):
                                    _, _ = model_lirpa.compute_bounds(x=(x_perturbed,),
                                                                      method='alpha-CROWN')
                                    save_dict = model_lirpa.save_intermediate()

                            lb, ub = save_dict.get('/6')

                            test_rs_loss, actual_n_unstable_nodes = compute_rs_loss(lb, ub, hidden_layer_dim)


                    running_test_loss += loss.item()
                    running_test_loss_1 += partial_loss_1
                    running_test_loss_2 += partial_loss_2

                    _, predicted = torch.max(outputs.data, 1)
                    total_test += targets.size(0)
                    correct_test += (predicted == targets).sum().item()

            test_loss = running_test_loss / len(test_loader)
            test_accuracy = 100 * correct_test / total_test
            loss_1_test = running_test_loss_1 / len(test_loader)
            loss_2_test = running_test_loss_2 / len(test_loader)

            print(f"  Test  -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

            if i_epoch == num_epochs - 1 :
                metrics.update({
                    'test_loss': test_loss, 'test_accuracy': test_accuracy,
                    'loss_on_test': loss_1_test, 'rs_loss_on_test': loss_2_test
                })

        print(f"Epoch {i_epoch + 1}:")
        print(f"  Train -> Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        if  i_epoch % REFINEMENT_FACTOR//2 == 0 or i_epoch == num_epochs - 1:
            print(f"  Test  -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%, Unstable nodes: {actual_n_unstable_nodes:.2f}")

        # Strategy for 7000 epochs
        if refinement_bool and (num_epochs > REFINEMENT_THRESHOLD) and (num_epochs % REFINEMENT_FACTOR == 0):
            if test_accuracy > previous_acc:
                if actual_n_unstable_nodes < previous_unstable_nodes:
                    i_epoch = num_epochs - 2
                    if DEBUG:
                        print("Best case scenario: training interrupted \n"
                              f"Nodi instabili:{actual_n_unstable_nodes:.2f} Prima: {previous_unstable_nodes:.2f}"
                              f"Accuracy: {test_accuracy:.2f}  Prima: {previous_acc:.2f}")
                else:

                    checkpoint = create_checkpoint(model_lirpa, optimizer, i_epoch)
                    if DEBUG:
                        print("CHECKPOINT CREATED"
                             f"Accuracy: {test_accuracy:.2f}  Prima: {previous_acc:.2f}")

                    if lambda_counter < LAMBDA_ITERATION_LIMIT:
                        old_rs_loss_factor = rs_loss_factor
                        rs_loss_factor += rs_loss_factor * LAMBDA_PERCENTAGE_INCREASING
                        metrics['lambda'].append(rs_loss_factor)
                        lambda_counter += 1
                        if DEBUG:
                            print(f"LAMBDA INCREASING number {lambda_counter}"
                                  f"Accuracy: {test_accuracy:.2f}  Prima: {previous_acc:.2f}"
                                  f"{old_rs_loss_factor=} {rs_loss_factor=}")


                    elif noise_counter < NOISE_ITERATION_LIMIT:
                        eps += eps * NOISE_PERCENTAGE_INCREASING
                        metrics['eps'].append(eps)
                        noise_counter += 1
                    else:
                        i_epoch = num_epochs - 2
            else:
                if checkpoint is not None:
                    set_epoch = load_checkpoint(model, optimizer, checkpoint, device)
                    i_epoch = set_epoch
                    metrics['num_epochs'] += 1
                    i_epoch = num_epochs - 2
                else:
                    i_epoch = num_epochs - 2

        i_epoch += 1

    return metrics, model_lirpa



if __name__ == '__main__':
    yaml_file = '../config_one_layered_full_dataset.yaml'
    config = load_yaml_config(yaml_file)
    hidden_dim = 100

    min_increment = 0
    max_increment = 5
    steps_limit = 10

    # Rs lambda for the smallest network, this values has to increase
    rs_factor = 0.1
    metrics, model  = generate_networks(config, rs_factor=rs_factor, hidden_dim=hidden_dim, calculate_epochs=calculate_epochs, dataset_name="MNIST")


if __name__ == '__main__':
    yaml_file = '../config_one_layered_full_dataset.yaml'
    config = load_yaml_config(yaml_file)

    min_increment = 0
    max_increment = 5
    steps_limit = 10

    # Rs lambda for the smallest network, this values has to increase
    rs_factor = 0.1

    for dataset in config['data']['dataset']:

        hidden_layer_dims = [50,  200, 500, 700, 900, 1100, 2000, 5000, 8000, 10000, 15000]

        OUTPUT_FOLDER = f"networks\\{dataset}"
        CSV_FILE = os.path.join(OUTPUT_FOLDER, "results.csv")

        # First model baseline
        first_model = hidden_layer_dims.pop(0)

        # Best accuracy for the smallest netowork. That value is gonna be the value to improve with over-parametrization
        metrics, model = generate_networks(config, rs_factor=rs_factor, hidden_dim=first_model,
                                           calculate_epochs=calculate_epochs, dataset_name=dataset)
        save_model(model, first_model, OUTPUT_FOLDER, device)
        write_results_on_csv(CSV_FILE, metrics)

        # Baseline accuracy
        previous_accuracy = metrics['test_accuracy']
        previous_unstable_nodes = None

        # Dict that collects all data of the networks that have improved the neurons stability of the NN
        best_models_dict = dict()

        for idx, h_dim in enumerate(hidden_layer_dims):
            print(f"Number of h_dim {hidden_layer_dims}")

            target_model = None
            target_metrics = None
            target_rs_loss = None
            old_weights = None


            # Increasing the lambda parameter of the RSLoss until it stops decreasing the accuracy
            min_increment = 0.1
            max_increment = 7
            increment = (max_increment - min_increment)/2

            steps_counter = 0
            failure_bool = True
            exception_counter = 0

            while steps_counter <= steps_limit:
                try:
                    metrics, model = generate_networks(config, rs_factor=rs_factor, hidden_dim=h_dim, calculate_epochs=calculate_epochs, dataset_name=dataset, to_beat_metric = get_min_index_and_value(best_models_dict[list(best_models_dict.keys())[-1]]))
                except Exception as e:
                    print("Exception: ", e)

                    if exception_counter == 3:
                        print("Experiment aborted: more that 3 exceptions launched")

                    print("Exception: ", e)
                    print("Trying again...")
                    exception_counter += 1
                    continue

                if metrics['test_accuracy'] + 0.006 >= previous_accuracy:
                    if not best_models_dict[str(h_dim)]:
                        best_models_dict[str(h_dim)] = list()

                    target_rs_loss = rs_factor + increment
                    min_increment = increment
                    increment = min_increment + (max_increment - min_increment)/2
                    best_models_dict[str(h_dim)].append((model, metrics))
                    failure_bool = False

                    if previous_unstable_nodes is not None:
                        if metrics['n_unstable_nodes'] < previous_unstable_nodes:
                            break

                elif metrics['test_accuracy'] < previous_accuracy:
                    max_increment = increment
                    increment = min_increment +(max_increment - min_increment)/2

                steps_counter += 1

                if min_increment > max_increment:
                    raise ValueError("min_increment > max_increment. Error in binary research implementation")

            if not failure_bool:
                model, metrics = get_min_index_and_value(best_models_dict[str(h_dim)])

                # Old and new (better accuracy) accuracy
                old_accuracy = previous_accuracy
                new_accuracy = metrics['test_accuracy']

                # Old and new (better accuracy) accuracy
                old_unstable_nodes = previous_unstable_nodes
                new_unstable_nodes = metrics['n_unstable_nodes']


                previous_accuracy = new_accuracy
                previous_unstable_nodes = new_unstable_nodes
                save_model(model, h_dim, OUTPUT_FOLDER, device)
                write_results_on_csv(CSV_FILE, metrics)
                rs_factor = target_rs_loss
                print(f"Accuracy of network with {h_dim} has set the accuracy minimum to {previous_accuracy}")

                print(f"{metrics=}")

            else:
                print(f"Network with {h_dim} filters has failed.")

            print(best_models_dict)