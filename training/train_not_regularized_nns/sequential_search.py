import os
import logging

from ..utils.dataset import get_data_loader
from ..utils.utils import load_yaml_config, save_models, write_results_on_csv
from training.train_not_regularized_nns import NUM_EPOCHS, RESULTS_FOLDER, CSV_FILE_ALL_CANDIDATES, CSV_FILE_BEST_CANDIDATES, device


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


class SequentialTraining:

    def __init__(self, model_cls, config_file_path, dataset_name, candidates_network_archs, train_batch_dim=128,
                 test_batch_dim=64):
        self.models = _generate_model(model_cls, candidates_network_archs)
        self.config = load_yaml_config(config_file_path)
        self.train_data_loader, self.test_data_loader, self.dummy_input, self.input_dim, self.output_dim = get_data_loader(dataset_name, train_batch_dim, test_batch_dim, input_flattened=False)

        self.metrics_collection = list()
        self.model_collection = list()
        self.save_folder_best_candidates = os.path.join(RESULTS_FOLDER, dataset_name, "best_models")
        self.csv_file_path_best_candidates = CSV_FILE_BEST_CANDIDATES
        self.device = device


        self.logger = logging.getLogger(__name__)
        self.logger.info("Device: %s", self.device)


    def sequential_training(self, trainer_manager, previous_model_weights=None, early_stopping=False):
        # Estrai il primo modello dalla lista
        first_model = self.models.pop(0)
        first_model_arch = first_model.get_shape()

        # Inizializza il manager di training per il primo modello
        baseline_trainer = trainer_manager(target_acc=0, data_loader=(self.train_data_loader, self.test_data_loader))

        # Allena il primo modello
        first_score, first_model = baseline_trainer.find_overfitted_best_model(
            first_model,
            first_model_arch,
            self.config,
            max_num_epochs=NUM_EPOCHS
        )

        # Salva il primo modello e registra il risultato
        save_models(first_model, first_model.identifier, self.save_folder_best_candidates, self.device,
                    self.dummy_input)
        write_results_on_csv(self.csv_file_path_best_candidates, first_score)

        previous_acc = first_score['test_accuracy']
        previous_model = first_model

        # Allena i modelli restanti sequenzialmente
        for model_ in self.models:
            model_shape = model_.get_shape()
            model_trainer = trainer_manager(target_acc=previous_acc,
                                            data_loader=(self.train_data_loader, self.test_data_loader))

            # Primo tentativo: riuso del modello precedente
            self.logger.info("Training new arch with Weight Reusing", model_.identifier)
            score, trained_model = model_trainer.find_overfitted_best_model(
                model_, model_shape, self.config, max_num_epochs=NUM_EPOCHS,
                previous_model=previous_model, early_stopping=early_stopping
            )

            # Se fallisce, riprova senza previous_model
            if score is None or trained_model is None:
                self.logger.info("Retrying without Weight Reusing", model_.identifier)
                score, trained_model = model_trainer.find_overfitted_best_model(
                    model_, model_shape, self.config, max_num_epochs=NUM_EPOCHS,
                    early_stopping=early_stopping
                )

            if score is not None and trained_model is not None:
                if score['test_accuracy'] > previous_acc:
                    self.logger.info(
                        "Model %s achieved better accuracy %.4f > %.4f. Saving model...",
                        trained_model.identifier,
                        score['test_accuracy'],
                        previous_acc
                    )
                    previous_acc = score['test_accuracy']
                    previous_model = trained_model

                    save_models(trained_model, trained_model.identifier,
                                self.save_folder_best_candidates, self.device, self.dummy_input)
                    write_results_on_csv(self.csv_file_path_best_candidates, score)
            else:
                self.logger.warning("Model %s failed or did not improve. Skipping...", model_.identifier)