import os
import logging

import torch
from torch import save

from training.regularized_trainer import ModelTrainingManager
from utils.dataset import get_dataset, get_data_loader, get_data_loader_testing
from utils.utils import load_yaml_config, write_results_on_csv, save_models
from training import REFINEMENT_PERCENTAGE, REFINEMENT_CYCLE_LENGTH, NUMBER_OF_CYCLES, NUM_EPOCHS, NOISE, \
    ACCURACY_THRESHOLD,RS_LOSS_FIRST_NN, RESULTS_FOLDER, CSV_FILE_ALL_CANDIDATES, CSV_FILE_BEST_CANDIDATES, BACKUP_FOLDER, device


def _generate_model(model_cls, candidates_network_archs):
    to_ret_list = list()
    for tuple_ in candidates_network_archs:
        model = model_cls(*tuple_)
        to_ret_list.append(model)

    return to_ret_list


def _get_min_index_and_value(results_dict):
    best_index, best_tuple = min(
        enumerate(results_dict),
        key=lambda x: x[1][2]['test_unstable_nodes']
    )
    return results_dict[best_index][0], results_dict[best_index][1], results_dict[best_index][2]


class BinaryHyperParamsResearch:

    def __init__(self, model_cls, config_file_path, dataset_name, candidates_network_archs, train_batch_dim=128,
                 test_batch_dim=64):
        self.models = _generate_model(model_cls, candidates_network_archs)
        self.config = load_yaml_config(config_file_path)
        self.train_data_loader, self.test_data_loader, self.dummy_input, self.input_dim, self.output_dim = get_data_loader(dataset_name, train_batch_dim, test_batch_dim, input_flattened=False)

        self.metrics_collection = list()
        self.model_collection = list()
        self.save_folder_best_candidates = os.path.join(RESULTS_FOLDER, dataset_name, "best_models")
        self.save_folder_all_candidates = os.path.join(RESULTS_FOLDER, dataset_name, "all_models")
        self.csv_file_path_best_candidates = CSV_FILE_BEST_CANDIDATES
        self.csv_file_path_all_candidates = CSV_FILE_ALL_CANDIDATES
        self.device = device

        self.logger = logging.getLogger(__name__)
        self.logger.info("Device: %s", self.device)

    def binary_search(self, limit_min_increment, limit_max_increment, steps_limit, trainer_manager):
        # Rs lambda for the smallest network, this values has to increase
        rs_factor = RS_LOSS_FIRST_NN

        # Get baseline model and metrics 
        first_model = self.models.pop(0)
        first_model_arch = first_model.get_shape()

        baseline_model_training_manager = trainer_manager(
            0,
            1000000,
            data_loader=(self.train_data_loader, self.test_data_loader),
            number_of_cycle=NUMBER_OF_CYCLES,
            refinement_percentage=REFINEMENT_PERCENTAGE,
            refinement_cycle_length=REFINEMENT_CYCLE_LENGTH,
        )

        baseline_metrics, baseline_model, baseline_model_ref = baseline_model_training_manager.train(first_model,
                                                                                                     first_model.get_shape(),
                                                                                                     self.dummy_input,
                                                                                                     self.config,
                                                                                                     num_epochs=NUM_EPOCHS,
                                                                                                     rsloss_lambda=RS_LOSS_FIRST_NN,
                                                                                                     eps=NOISE)

        # Save baseline results
        save_models(baseline_model_ref, baseline_model_ref.identifier, self.save_folder_best_candidates, self.device, self.dummy_input)
        write_results_on_csv(self.csv_file_path_best_candidates, baseline_metrics)

        self.logger.info("Minimum Accuracy and Unstable Nodes threshold set by baseline model's results with architecture %s:", first_model_arch)

        # Initialize tracking variables
        previous_accuracy = baseline_metrics['test_accuracy']
        previous_unstable_nodes = baseline_metrics['test_unstable_nodes']
        best_models_dict = {baseline_model_ref.identifier: [
            (baseline_model, baseline_model_ref, baseline_metrics)]}
        print(best_models_dict.keys())# Collects models that improved stability

        # Evaluate remaining models
        for idx, model_untr in enumerate(self.models):
            self.logger.info("Training and refining the %d-th model", idx)

            # Initialize search parameters
            min_increment = limit_min_increment
            max_increment = limit_max_increment

            target_rs_loss = rs_factor
            increment = (max_increment - min_increment) / 2
            steps_counter = 0
            failure_bool = True

            # Binary search for optimal rs_factor
            while steps_counter <= steps_limit:
                # Metrica da battere del precedente modello di dimensione inferiore andato a buon fine
                to_beat_metric = _get_min_index_and_value(best_models_dict[list(best_models_dict.keys())[-1]])
                print(to_beat_metric)

                self.logger.info("Iteration %d", steps_counter)

                # Train model with current parameters
                model_training_manager = trainer_manager(
                    previous_accuracy,
                    previous_unstable_nodes,
                    data_loader=(self.train_data_loader, self.test_data_loader),
                    number_of_cycle=NUMBER_OF_CYCLES,
                    refinement_percentage=REFINEMENT_PERCENTAGE,
                    refinement_cycle_length=REFINEMENT_CYCLE_LENGTH,

                )

                metrics, model, model_ref = model_training_manager.train(
                    model_untr,
                    model_untr.get_shape(),
                    self.dummy_input,
                    self.config,
                    num_epochs=NUM_EPOCHS,
                    rsloss_lambda=rs_factor + increment,
                    eps=NOISE
                )

                if metrics['test_unstable_nodes'] > to_beat_metric[2]['test_unstable_nodes']:
                    self.logger.info("Model %d needs refinement - current unstable nodes: %f --- Refining attempt",
                                   idx, metrics['test_unstable_nodes'])
                    # Save model state dict
                    backup_temp_model_path = os.path.join(BACKUP_FOLDER, 'model_' + str(idx) + '_state.pth')
                    save(model_ref.state_dict(),
                         backup_temp_model_path)
                    success_flag, refined_metrics, refined_model, refined_ref_model = model_training_manager.refinement_training(
                        model_untr,
                        model_untr.get_shape(),
                        self.dummy_input,
                        self.config,
                        initial_rsloss_lambda=rs_factor + increment,
                        eps=NOISE,
                        model_path=backup_temp_model_path
                    )

                    if success_flag:
                        if refined_metrics['test_unstable_nodes'] < metrics['test_unstable_nodes']:
                            self.logger.info("Refinement successful for model %d", idx)
                            model = refined_model
                            model_ref = refined_ref_model
                            metrics = refined_metrics

                # Saving the model whatever its performances
                save_models(model_ref, model_ref.identifier + '_'+ str(metrics['lambda']), self.save_folder_all_candidates, self.device, self.dummy_input)
                write_results_on_csv(self.csv_file_path_all_candidates, metrics)

                # Check if model improved accuracy
                if metrics['test_accuracy'] + ACCURACY_THRESHOLD >= previous_accuracy:
                    if str(idx) not in best_models_dict:
                        best_models_dict[str(idx)] = []

                    # Update parameters
                    target_rs_loss = rs_factor + increment
                    min_increment = increment
                    increment = min_increment + (max_increment - min_increment) / 2

                    # Save a successful model
                    best_models_dict[str(idx)].append((model, model_ref, metrics))
                    failure_bool = False

                    # Check if stability improved
                    if previous_unstable_nodes is not None:
                        if metrics['test_unstable_nodes'] < previous_unstable_nodes:
                            self.logger.info("Model %d achieved better stability - stopping search", idx)
                            break

                else:  # Accuracy decreased
                    max_increment = increment
                    increment = min_increment + (max_increment - min_increment) / 2

                steps_counter += 1

                if min_increment > max_increment:
                    self.logger.error("min_increment > max_increment. Error in binary research implementation")
                    raise ValueError("min_increment > max_increment. Error in binary research implementation")

            if not failure_bool:
                # Get best model for this architecture
                model, model_ref, metrics = _get_min_index_and_value(best_models_dict[str(idx)])

                # Update tracking metrics
                previous_accuracy = metrics['test_accuracy']
                previous_unstable_nodes = metrics['test_unstable_nodes']

                # Save results
                save_models(model_ref, model_ref.identifier, self.save_folder_best_candidates, self.device, self.dummy_input)
                write_results_on_csv(CSV_FILE_BEST_CANDIDATES, metrics)
                rs_factor = target_rs_loss

                self.logger.info("Accuracy of network with %d has set the accuracy minimum to %f", idx, previous_accuracy)


            else:
                self.logger.info("Network with %d filters has failed.", idx)