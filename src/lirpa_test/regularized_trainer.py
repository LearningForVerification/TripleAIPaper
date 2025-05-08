import copy
import logging
from os import PathLike
from typing import Tuple, Any, Union
from abc import ABC, abstractmethod

import numpy as np
import torch
from auto_LiRPA import PerturbationLpNorm, BoundedTensor, BoundedModule
from torch import Tensor
from torch import optim, multiprocessing
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, Type
from src.lirpa_test import VALIDATION_FREQUENCY, ACCURACY_THRESHOLD, DEBUG


class ModelTrainingManager(ABC):
    def __init__(self, target_acc: float, inst_target: float,
                 data_loader: Tuple[DataLoader, DataLoader], number_of_cycle: int, refinement_percentage: float, refinement_cycle_length: int,
                 verbose: bool = True
                 ):
        self.logger = logging.getLogger(__name__)

        # The accuracy that must be beaten
        self.target_accuracy = target_acc

        # Instability target
        self.instability_target = inst_target

        # DataLoader of the dataset
        self.train_data_loader, self.test_loader = data_loader

        # The best model found: Must be returned
        self.best_found_model = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.number_of_cycle = number_of_cycle

        self.refinement_percentage = refinement_percentage

        self.refinement_cycle_length = refinement_cycle_length

    @abstractmethod
    def get_rsloss(self, model: nn.Module, model_ref, architecture_tuple: tuple, input_batch: Tensor,
                   perturbation: PerturbationLpNorm, eps: float, method='ibp') -> Tuple[Tensor, Dict]:
        """
        Calculate the regularization loss for the model.

        Args:
            model: Neural network model
            architecture_tuple: Tuple describing network architecture 
            input_batch: Input tensor batch
            perturbation: Perturbation configuration
            method: Method to use for bound computation (default: 'ibp')

        Returns:
            Tuple containing:
            - Tensor representing the regularization loss
            - Dict containing bound information
        """
        pass

    def train(self,
              model_untr: nn.Module,
              arch_tuple: tuple,
              dummy_input: Tensor,
              data_dict: Dict[str, Any],
              num_epochs: int,
              rsloss_lambda: float,
              eps: float = None
              ) -> tuple[dict[str | Any, float | None | Any], BoundedModule, Module]:

        model_ref = copy.deepcopy(model_untr)
        model_ref.to(device=self.device)
        model = BoundedModule(model_ref, dummy_input, device=self.device)


        """Same docstring but updated for single training mode"""
        # Validate required data_dict structure
        required_keys = ['optimizer', 'scheduler_lr', 'data', 'training']
        if not all(key in data_dict for key in required_keys):
            raise ValueError(f"data_dict missing required keys: {required_keys}")

        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

        if rsloss_lambda < 0:
            raise ValueError("rsloss_lambda must be non-negative")

        if eps is not None and eps <= 0:
            raise ValueError("eps must be positive")

        # Unpack data_dict
        optimizer_dict = data_dict['optimizer']
        scheduler_lr_dict = data_dict['scheduler_lr']

        # Create optimizer params dict
        opt_params = optimizer_dict.copy()
        optimizer_name = opt_params['type']
        del opt_params['type']

        # NN architectures
        input_dim = int(data_dict['data']['input_dim'])
        output_dim = int(data_dict['data']['output_dim'])
        hidden_dim = data_dict['data'].get('hidden_dim')

        loss_name = data_dict['training']['loss_name']
        num_classes = int(data_dict['training'].get('num_classes', output_dim))

        train_unstable_nodes = 0

        # Define the optimizer function
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

        #try:
        optimizer = optimizer_cls(model.parameters(), **opt_params)
        criterion = criterion_cls()

        # Training the model
        for epoch in range(num_epochs):
            self.logger.debug(f"Epoch {epoch + 1}/{num_epochs}")
            self._train_epoch_and_get_stats(
                model=model,
                model_ref = model_ref,
                device=self.device,
                arch_tuple=arch_tuple,
                optimizer=optimizer,
                criterion=criterion,
                num_classes=num_classes,
                rsloss_lambda=rsloss_lambda,
                eps=eps
            )

            if epoch % VALIDATION_FREQUENCY == 0 and epoch != 0:
                self.logger.debug(f"Evaluating model on test set at epoch {epoch + 1}/{num_epochs}")
                test_accuracy, _, _, _, test_unstable_nodes = self.calculate_accuracy_and_loss(
                    model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda,
                    train_set=False, eps=eps)


        # Calculate final metrics
        test_accuracy, test_loss, partial_loss_test, partial_rsloss_test, test_unstable_nodes = self.calculate_accuracy_and_loss(
            model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=False, eps=eps)
        train_accuracy, train_loss, partial_loss_train, partial_rsloss_train, train_unstable_nodes = self.calculate_accuracy_and_loss(
            model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=True, eps=eps)

        # TEST that the original nn.Module has the same 
        model_ref_bounded = BoundedModule(model_ref, dummy_input, device=self.device)
        test_accuracy__DEBUG, test_loss__DEBUG, partial_loss_test__DEBUG, partial_rsloss_test__DEBUG, test_unstable_nodes__DEBUG = self.calculate_accuracy_and_loss(
            model_ref_bounded, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=False,
            eps=eps)
        train_accuracy__DEBUG, train_loss__DEBUG, partial_loss_train__DEBUG, partial_rsloss_train__DEBUG, train_unstable_nodes__DEBUG = self.calculate_accuracy_and_loss(
            model_ref_bounded, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=True, eps=eps)

        # Compare results between model and model_ref using assertions
        assert abs(
            test_accuracy - test_accuracy__DEBUG) < 1e-3, f"Test accuracy mismatch: {test_accuracy:.2f} vs {test_accuracy__DEBUG:.2f}"
        assert abs(
            train_accuracy - train_accuracy__DEBUG) < 1e-3, f"Train accuracy mismatch: {train_accuracy:.2f} vs {train_accuracy__DEBUG:.2f}"
  
        score={
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'partial_loss_train': partial_loss_train,
            'partial_loss_test': partial_loss_test,
            'rs_train_loss': partial_rsloss_train,
            'rs_test_loss': partial_rsloss_test,
            'lambda': rsloss_lambda,
            'cycle': 1,
            'eps': eps,
            'train_unstable_nodes': train_unstable_nodes,
            'test_unstable_nodes': test_unstable_nodes,
            'architecture' : arch_tuple,
        }


        
        self.logger.info(f"Training completed with architecture {arch_tuple} and rsloss_lambda {rsloss_lambda} with following metrics:")
        self.logger.info(f"Train accuracy: {train_accuracy:.2f}%")
        self.logger.info(f"Test accuracy: {test_accuracy:.2f}%")
        self.logger.info(f"Train loss: {train_loss:.4f}")
        self.logger.info(f"Test loss: {test_loss:.4f}")
        self.logger.info(f"Train unstable nodes: {train_unstable_nodes}")
        self.logger.info(f"Test unstable nodes: {test_unstable_nodes}")

        return score, model, model_ref

        # except Exception as e:
        #     raise RuntimeError(f"Training failed: {str(e)}") from e

    def refinement_training(self,
                            model_untr: nn.Module,
                            arch_tuple: tuple,
                            dummy_input:Tensor,
                            data_dict: Dict[str, Any],
                            initial_rsloss_lambda: float,
                            eps: float,
                            model_path: PathLike
                            ) -> bool | tuple[
        bool, dict[str | Any, float | int | Any], BoundedModule | Module, BoundedModule | Module] | tuple[
                                     bool, None, None, None]:

        model_ref = copy.deepcopy(model_untr)
        if not model_path:
            raise ValueError("model_path must be provided for refinement training")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if initial_rsloss_lambda < 0:
            raise ValueError("initial_rsloss_lambda must be non-negative")

        self.logger.info(
            f"NETWORK ARCHITECTURE: {arch_tuple} with initial_rsloss_lambda={initial_rsloss_lambda} and eps={eps}")

        # try:
        # Load pretrained model
        model_ref.load_state_dict(torch.load(model_path, map_location=self.device))
        model = BoundedModule(model_ref, dummy_input,
                              device=self.device)

        # Start with initial lambda
        rsloss_lambda = initial_rsloss_lambda * (1 + self.refinement_percentage)
        success = False
        #
    # try:
        # Setup training parameters
        # Unpack data_dict
        optimizer_dict = data_dict['optimizer']
        opt_params = optimizer_dict.copy()
        optimizer_name = opt_params['type']
        del opt_params['type']

        # Get number of classes
        num_classes = int(data_dict['training'].get('num_classes',
                                                    int(data_dict['data']['output_dim'])))

        # Get loss function
        loss_name = data_dict['training']['loss_name']

        # Define optimizer class
        if optimizer_name == 'Adam':
            optimizer_cls = optim.Adam
        elif optimizer_name == 'SGD':
            optimizer_cls = optim.SGD
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Define loss function class
        if loss_name == 'CrossEntropyLoss':
            criterion_cls = nn.CrossEntropyLoss
        elif loss_name == 'MSE':
            criterion_cls = nn.MSELoss
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

        optimizer = optimizer_cls(model.parameters(), **opt_params)
        criterion = criterion_cls()
        backup_model = (None, None)
        cycle_counter = cycle_counter_backup = 1

        # Refinement training loop
        for epoch in range(self.number_of_cycle * self.refinement_cycle_length):
            self._train_epoch_and_get_stats(
                model=model,
                model_ref = model_ref,
                device=self.device,
                arch_tuple=arch_tuple,
                optimizer=optimizer,
                criterion=criterion,
                num_classes=num_classes,
                rsloss_lambda=rsloss_lambda,
                eps=eps
            )
            if epoch % self.refinement_cycle_length == 0 and epoch != 0:
                test_accuracy, _, _, _, test_unstable_nodes = self.calculate_accuracy_and_loss(
                    model, model_ref, arch_tuple, criterion, num_classes,
                    rsloss_lambda=rsloss_lambda, train_set=False, eps=eps)

                if test_accuracy >= self.target_accuracy + ACCURACY_THRESHOLD:
                    if test_unstable_nodes <= self.instability_target:
                        self.logger.info(
                            f"Refinement successful: {test_unstable_nodes=} < {self.instability_target=} and "
                            f"{test_accuracy=} > {self.target_accuracy=}")

                        success = True
                        break


                    self.logger.info(
                        f"Creating backup with {test_accuracy=}, {test_unstable_nodes=}, {rsloss_lambda=}")
                    backup_model_ref = copy.deepcopy(model_ref)
                    backup_model = BoundedModule(backup_model_ref, dummy_input, device=self.device)
                    backup_model = (backup_model, backup_model_ref)
                    cycle_counter_backup = cycle_counter
                    rsloss_lambda *= (1 + self.refinement_percentage)
                    cycle_counter += 1
                else:
                    if backup_model[0] is None:
                        self.logger.info("Backup not available, refinement failed")
                        return  False, None, None, None

                    self.logger.info("Restoring from last successful backup")
                    model = backup_model[0]
                    model_ref = backup_model[1]
                    cycle_counter = cycle_counter_backup
                    success = True
                    break
        if success:
            # Calculate final metrics
            test_accuracy, test_loss, partial_loss_test, partial_rsloss_test, test_unstable_nodes = self.calculate_accuracy_and_loss(
                model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=False,
                eps=eps)
            train_accuracy, train_loss, partial_loss_train, partial_rsloss_train, train_unstable_nodes = self.calculate_accuracy_and_loss(
                model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=True, eps=eps)

            score = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'partial_loss_train': partial_loss_train,
                'partial_loss_test': partial_loss_test,
                'rs_train_loss': partial_rsloss_train,
                'rs_test_loss': partial_rsloss_test,
                'lambda': rsloss_lambda,
                'cycle': cycle_counter,
                'eps': eps,
                'train_unstable_nodes': train_unstable_nodes,
                'test_unstable_nodes': test_unstable_nodes,
                'architecture': arch_tuple,
            }

            self.logger.info(
                f"Final eps={eps}")
            self.logger.info("Training completed with following metrics:")
            self.logger.info(f"Train accuracy: {train_accuracy:.2f}%")
            self.logger.info(f"Test accuracy: {test_accuracy:.2f}%")
            self.logger.info(f"Train loss: {train_loss:.4f}")
            self.logger.info(f"Test loss: {test_loss:.4f}")
            self.logger.info(f"Train unstable nodes: {train_unstable_nodes}")
            self.logger.info(f"Test unstable nodes: {test_unstable_nodes}")
            return True, score, model, model_ref
        else:
            self.logger.info("Refinement failed")
            return  False, None, None, None
        #
        #     except Exception as e:
        #         raise RuntimeError(f"Refinement training failed: {str(e)}") from e
        #
        # except Exception as e:
        #     raise RuntimeError(f"Refinement training failed: {str(e)}") from e

    def _calculate_loss(
            self,
            model: nn.Module,
            model_ref: nn.Module,
            architecture_tuple: tuple,
            loss_criterion: nn.Module,
            num_classes: int,
            rsloss_lambda: float = None,
            train_set: bool = False,
            eps: float = 0.015
    ) -> tuple[float | Any, float | Any, float | Any, int | Any] | float | Any:
        """Calculate loss values for model evaluation"""
        if not hasattr(self, 'train_data_loader') or not hasattr(self, 'test_loader'):
            self.logger.info("Data loaders not initialized")
            raise AttributeError("Data loaders not initialized")

        if next(model.parameters()).device != self.device:
            raise ValueError(f"Model must be on {self.device}")

        model.eval()
        running_loss = 0.0
        rs_loss = 0.0
        perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)
        unstable_nodes = 0

        data_loader = self.train_data_loader if train_set else self.test_loader
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device).to(torch.float32), targets.to(self.device)
                targets.long()
                loss = None
                outputs = None
                outputs = model(inputs)

                if isinstance(loss_criterion, nn.MSELoss):
                    targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                    loss = loss_criterion(outputs, targets_hot_encoded)
                else:
                    loss: Tensor = loss_criterion(outputs, targets)
                    running_loss += loss.item()
                if rsloss_lambda is not None:
                    _rs_loss, _unstable_nodes = self.get_rsloss(model=model, model_ref=model_ref, architecture_tuple=architecture_tuple,
                                               input_batch=inputs, perturbation=perturbation, eps=eps)
                    rs_loss += _rs_loss.item()
                    unstable_nodes += _unstable_nodes
                
                del outputs
                del loss

                if 'targets_hot_encoded' in locals():
                    del targets_hot_encoded
                torch.cuda.empty_cache()

        partial_loss = running_loss / len(data_loader)
        if rsloss_lambda is not None:
            partial_rs_loss = rs_loss
            total_loss = partial_loss + rsloss_lambda * partial_rs_loss
            unstable_nodes = unstable_nodes / len(data_loader)
            return total_loss, partial_loss, partial_rs_loss, unstable_nodes
        else:
            return partial_loss

    def _calculate_accuracy(
            self,
            model: nn.Module,
            train_set: bool = False
    ) -> float:
        """Calculate accuracy for model evaluation"""
        if not hasattr(self, 'train_data_loader') or not hasattr(self, 'test_loader'):
            self.logger.info("Data loaders not initialized")
            raise AttributeError("Data loaders not initialized")

        if next(model.parameters()).device != self.device:
            raise ValueError(f"Model must be on {self.device}")

        model.eval()
        correct_count, total_count = 0, 0

        try:
            data_loader = self.train_data_loader if train_set else self.test_loader
            with torch.no_grad():
                for inputs, targets in data_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_count += targets.size(0)
                    correct_count += (predicted == targets).sum().item()

            accuracy = 100 * correct_count / total_count
            return accuracy

        except RuntimeError as e:
            raise RuntimeError(f"Error calculating accuracy: {str(e)}") from e

    def calculate_accuracy_and_loss(
            self,
            model: nn.Module,
            model_ref: nn.Module,
            arch_tuple: tuple,
            loss_criterion: nn.Module,
            num_classes: int,
            rsloss_lambda: float,
            train_set: bool = False,
            eps: float = 0.015) -> tuple:
        """Calculate accuracy and loss for either training or test dataset"""
        
        if rsloss_lambda is not None:
            total_loss, partial_loss, partial_rs_loss, unstable_nodes = self._calculate_loss(model, model_ref, arch_tuple, loss_criterion, num_classes, rsloss_lambda, train_set, eps)
        else:
            total_loss = self._calculate_loss(model, model_ref, arch_tuple, loss_criterion, num_classes, train_set=train_set)

        accuracy = self._calculate_accuracy(model, train_set)

        if hasattr(self, 'verbose') and self.verbose:
            set_name = "Training" if train_set else "Test"
            self.logger.info(f"Statistics on {set_name} Set:")
            if rsloss_lambda is not None:
                self.logger.info(f"  Total Loss: {total_loss:.4f}, Base Loss: {partial_loss:.4f}, "
                      f"RS Loss: {partial_rs_loss:.4f}, Accuracy: {accuracy:.2f}%")
            else:
                self.logger.info(f"  Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if rsloss_lambda is not None:
            return accuracy, total_loss, partial_loss, partial_rs_loss, unstable_nodes
        else:
            return accuracy, total_loss

    def _train_epoch_and_get_stats(self, model, model_ref, device, arch_tuple, optimizer, criterion, num_classes, rsloss_lambda, eps):
        model.train()
        running_train_loss = running_train_loss_1 = running_train_loss_2 = 0.0
        correct_train = total_train = 0
        perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)
        train_unstable_nodes = 0

        for index, (inputs, targets) in enumerate(self.train_data_loader):
            inputs, targets = inputs.to(device).to(torch.float32), targets.to(device)
            targets = targets.long()
            outputs = model(inputs)

            # Calculate losses
            if isinstance(criterion, nn.MSELoss):
                targets_hot = F.one_hot(targets, num_classes=num_classes).float()
                loss = criterion(outputs, targets_hot)
            else:
                loss = criterion(outputs, targets)
            partial_loss_1 = loss.item()
            rsloss_result, unstable_nodes = self.get_rsloss(model, model_ref, arch_tuple, inputs, perturbation, eps)
            partial_loss_2 = rsloss_result
            train_unstable_nodes += unstable_nodes
            total_loss = loss + rsloss_lambda * partial_loss_2

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track metrics
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            running_train_loss += total_loss.item()
            running_train_loss_1 += partial_loss_1
            running_train_loss_2 += partial_loss_2.item()

        # Calculate epoch statistics
        dataset_size = len(self.train_data_loader)
        epoch_stats = {
            'train_loss': running_train_loss / dataset_size,
            'train_accuracy': 100 * correct_train / total_train,
            'loss_1_train': running_train_loss_1 / dataset_size,
            'loss_2_train': running_train_loss_2
        }

        if DEBUG:
            self.logger.info(f"Epoch Statistics:")
            self.logger.info(f"  Train -> Loss: {epoch_stats['train_loss']:.4f}, "
                  f"Accuracy: {epoch_stats['train_accuracy']:.2f}%")

        return train_unstable_nodes/len(self.train_data_loader)