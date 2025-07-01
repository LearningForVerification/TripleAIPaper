import copy
import logging
from abc import ABC, abstractmethod
from os import PathLike
from typing import Dict
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import PerturbationLpNorm, BoundedModule
from torch import Tensor
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader

from training import VALIDATION_FREQUENCY, ACCURACY_THRESHOLD, LAMBDA_LR_CYCLE, CONV_RSLOSS_LAMBDA_MULTIPLIER, DEBUG


class ModelTrainingManager(ABC):
    def __init__(self, target_acc: float, inst_target: float,
                 data_loader: Tuple[DataLoader, DataLoader], number_of_cycle: int, refinement_percentage: float, refinement_cycle_length: int,
                 verbose: bool = True
                 ):
        self.logger = logging.getLogger(__name__)
        self.target_accuracy = target_acc
        self.instability_target = inst_target
        self.train_data_loader, self.test_loader = data_loader
        self.best_found_model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.number_of_cycle = number_of_cycle
        self.refinement_percentage = refinement_percentage
        self.refinement_cycle_length = refinement_cycle_length

    @abstractmethod
    def get_rsloss(self, model: nn.Module, model_ref, architecture_tuple: tuple, input_batch: Tensor,
                   perturbation: PerturbationLpNorm, eps: float, method='ibp'):
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
        output_dim = int(data_dict['data']['output_dim'])

        loss_name = data_dict['training']['loss_name']
        num_classes = int(data_dict['training'].get('num_classes', output_dim))

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

        # Fattore di decadimento: -1% ogni 400 epoche
        lambda_lr = lambda epoch: 0.99 ** (epoch // LAMBDA_LR_CYCLE)

        # Classe dello scheduler
        scheduler_lr_cls = torch.optim.lr_scheduler.LambdaLR
        scheduler_lr_params = {'lr_lambda': lambda_lr}

        # Inizializzare lo scheduler per il tasso di apprendimento
        scheduler = scheduler_lr_cls(optimizer, **scheduler_lr_params)

        # Training the model
        for epoch in range(num_epochs):
            self.logger.debug("Epoch %d/%d" % (epoch + 1, num_epochs))
            self._train_epoch_and_get_stats(model=model, model_ref=model_ref, device=self.device, arch_tuple=arch_tuple,
                                            optimizer=optimizer, criterion=criterion, num_classes=num_classes,
                                            rsloss_lambda=rsloss_lambda, eps=eps, scheduler=scheduler)

            if epoch % VALIDATION_FREQUENCY == 0 and epoch != 0:
                self.logger.debug("Evaluating model on test set at epoch %d/%d" % (epoch + 1, num_epochs))
                test_accuracy, _, _, _, test_unstable_nodes = self.calculate_accuracy_and_loss(
                    model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda,
                    train_set=False, eps=eps)

        # Calculate final metrics
        test_accuracy, test_loss, partial_loss_test, partial_rsloss_test_conv, partial_rsloss_test_fc, test_unstable_nodes = self.calculate_accuracy_and_loss(
            model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=False,
            eps=eps)
        train_accuracy, train_loss, partial_loss_train, partial_rsloss_train_conv, partial_rsloss_train_fc, train_unstable_nodes = self.calculate_accuracy_and_loss(
            model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=True,
            eps=eps)

        # Test that the bounded and unbounded models produce similar results
        model_ref_bounded = BoundedModule(model_ref, dummy_input, device=self.device)
        test_accuracy__DEBUG, test_loss__DEBUG, partial_loss_test__DEBUG, partial_rsloss_test_conv__DEBUG, partial_rsloss_test_fc__DEBUG, test_unstable_nodes__DEBUG = self.calculate_accuracy_and_loss(
            model_ref_bounded, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda,
            train_set=False,
            eps=eps)
        train_accuracy__DEBUG, train_loss__DEBUG, partial_loss_train__DEBUG, partial_rsloss_train_conv__DEBUG, partial_rsloss_train_fc__DEBUG, train_unstable_nodes__DEBUG = self.calculate_accuracy_and_loss(
            model_ref_bounded, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda,
            train_set=True, eps=eps)

        # Compare results between model and model_ref using assertions
        assert abs(
            test_accuracy - test_accuracy__DEBUG) < 1e-3, "Test accuracy mismatch: %.2f vs %.2f" % (test_accuracy, test_accuracy__DEBUG)
        assert abs(
            train_accuracy - train_accuracy__DEBUG) < 1e-3, "Train accuracy mismatch: %.2f vs %.2f" % (train_accuracy, train_accuracy__DEBUG)

        score = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'partial_loss_train': partial_loss_train,
            'partial_loss_test': partial_loss_test,
            'rs_train_loss_conv': partial_rsloss_train_conv,
            'rs_train_loss_fc': partial_rsloss_train_fc,
            'rs_test_loss_conv': partial_rsloss_test_conv,
            'rs_test_loss_fc': partial_rsloss_test_fc,
            'lambda': rsloss_lambda,
            'cycle': 0,
            'eps': eps,
            'train_unstable_nodes': train_unstable_nodes,
            'test_unstable_nodes': test_unstable_nodes,
            'architecture': arch_tuple,
        }


        
        self.logger.info("Training completed with architecture %s and rsloss_lambda %s with following metrics:" % (arch_tuple, rsloss_lambda))
        self.logger.info("Train accuracy: %.2f%%" % train_accuracy)
        self.logger.info("Test accuracy: %.2f%%" % test_accuracy)
        self.logger.info("Train loss: %.4f" % train_loss)
        self.logger.info("Test loss: %.4f" % test_loss)
        self.logger.info("Train unstable nodes: %s" % train_unstable_nodes)
        self.logger.info("Test unstable nodes: %s" % test_unstable_nodes)

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
                            ):

        model_ref = copy.deepcopy(model_untr)
        if not model_path:
            raise ValueError("model_path must be provided for refinement training")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if initial_rsloss_lambda < 0:
            raise ValueError("initial_rsloss_lambda must be non-negative")

        self.logger.info("NETWORK ARCHITECTURE: %s with initial_rsloss_lambda=%s and eps=%s" % 
                    (arch_tuple, initial_rsloss_lambda, eps))


        # Load pretrained model
        model_ref.load_state_dict(torch.load(model_path, map_location=self.device))
        model = BoundedModule(model_ref, dummy_input,
                              device=self.device)


        # Start with the initial lambda
        rsloss_lambda = initial_rsloss_lambda * (1 + self.refinement_percentage)
        success = False

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
            self._train_epoch_and_get_stats(model=model, model_ref=model_ref, device=self.device, arch_tuple=arch_tuple,
                                            optimizer=optimizer, criterion=criterion, num_classes=num_classes,
                                            rsloss_lambda=rsloss_lambda, eps=eps)
            if epoch % self.refinement_cycle_length == 0 and epoch != 0:
                test_accuracy, _, _, _, _, test_unstable_nodes = self.calculate_accuracy_and_loss(
                    model, model_ref, arch_tuple, criterion, num_classes,
                    rsloss_lambda=rsloss_lambda, train_set=False, eps=eps)

                if test_accuracy >= self.target_accuracy + ACCURACY_THRESHOLD:
                    if test_unstable_nodes <= self.instability_target:
                        self.logger.info("Refinement successful: test_unstable_nodes=%s < instability_target=%s and test_accuracy=%s > target_accuracy=%s" % 
                           (test_unstable_nodes, self.instability_target, test_accuracy, self.target_accuracy))

                        success = True
                        break


                    self.logger.info("Creating backup with test_accuracy=%s, test_unstable_nodes=%s, rsloss_lambda=%s" % 
                        (test_accuracy, test_unstable_nodes, rsloss_lambda))
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
            test_accuracy, test_loss, partial_loss_test, partial_rsloss_test_conv, partial_rsloss_test_fc, test_unstable_nodes = self.calculate_accuracy_and_loss(
                model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=False,
                eps=eps)
            train_accuracy, train_loss, partial_loss_train, partial_rsloss_train_conv, partial_rsloss_train_fc, train_unstable_nodes = self.calculate_accuracy_and_loss(
                model, model_ref, arch_tuple, criterion, num_classes, rsloss_lambda=rsloss_lambda, train_set=True, eps=eps)

            score = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'partial_loss_train': partial_loss_train,
                'partial_loss_test': partial_loss_test,
                'rs_train_loss_conv': partial_rsloss_train_conv,
                'rs_train_loss_fc': partial_rsloss_train_fc,
                'rs_test_loss_conv': partial_rsloss_test_conv,
                'rs_test_loss_fc': partial_rsloss_test_fc,
                'lambda': rsloss_lambda,
                'cycle': cycle_counter,
                'eps': eps,
                'train_unstable_nodes': train_unstable_nodes,
                'test_unstable_nodes': test_unstable_nodes,
                'architecture': arch_tuple,
            }

            self.logger.info("Final eps=%s" % eps)
            self.logger.info("Training completed with following metrics:")
            self.logger.info("Train accuracy: %.2f%%" % train_accuracy)
            self.logger.info("Test accuracy: %.2f%%" % test_accuracy)
            self.logger.info("Train loss: %.4f" % train_loss)
            self.logger.info("Test loss: %.4f" % test_loss)
            self.logger.info("Train unstable nodes: %s" % train_unstable_nodes)
            self.logger.info("Test unstable nodes: %s" % test_unstable_nodes)
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
    ):
        """Calculate loss values for model evaluation"""
        if not hasattr(self, 'train_data_loader') or not hasattr(self, 'test_loader'):
            self.logger.info("Data loaders not initialized")
            raise AttributeError("Data loaders not initialized")

        if next(model.parameters()).device != self.device:
            raise ValueError("Model must be on %s" % self.device)

        model.eval()
        running_loss = 0.0
        rs_loss_CONV = 0.0
        rs_loss_FC = 0.0
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
                    _rs_loss_CONV, _rs_loss_FC, _unstable_nodes = self.get_rsloss(model=model, model_ref=model_ref, architecture_tuple=architecture_tuple,
                                               input_batch=inputs, perturbation=perturbation, eps=eps)
                    rs_loss_CONV += _rs_loss_CONV.item()
                    rs_loss_FC += _rs_loss_FC.item()
                    unstable_nodes += _unstable_nodes
                
                del outputs
                del loss

                if 'targets_hot_encoded' in locals():
                    del targets_hot_encoded
                torch.cuda.empty_cache()

        partial_loss = running_loss / len(data_loader)
        if rsloss_lambda is not None:
            partial_rs_loss = rs_loss_CONV + rs_loss_FC
            total_loss = partial_loss + CONV_RSLOSS_LAMBDA_MULTIPLIER * rsloss_lambda * rs_loss_CONV + rsloss_lambda * rs_loss_FC
            unstable_nodes = unstable_nodes / len(data_loader)
            return total_loss, partial_loss, rs_loss_CONV, rs_loss_FC, unstable_nodes
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
            total_loss, partial_loss, rs_loss_conv, rs_loss_fc, unstable_nodes = self._calculate_loss(model, model_ref, arch_tuple, loss_criterion, num_classes, rsloss_lambda, train_set, eps)
        else:
            total_loss = self._calculate_loss(model, model_ref, arch_tuple, loss_criterion, num_classes, train_set=train_set)

        accuracy = self._calculate_accuracy(model, train_set)

        if hasattr(self, 'verbose') and self.verbose:
            set_name = "Training" if train_set else "Test"
            self.logger.info("Statistics on %s Set:" % set_name)
            if rsloss_lambda is not None:
                self.logger.info("  Total Loss: %.4f, Base Loss: %.4f, RS Loss CONV: %.4f, RS Loss FC: %.4f, Accuracy: %.2f%%" %
                           (total_loss, partial_loss, rs_loss_conv, rs_loss_fc, accuracy))
            else:
                self.logger.info("  Loss: %.4f, Accuracy: %.2f%%" % (total_loss, accuracy))

        if rsloss_lambda is not None:
            return accuracy, total_loss, partial_loss, rs_loss_conv, rs_loss_fc, unstable_nodes
        else:
            return accuracy, total_loss

    def _train_epoch_and_get_stats(self, model, model_ref, device, arch_tuple, optimizer, criterion, num_classes,
                                   rsloss_lambda, eps, scheduler=None):
        model.train()

        """Enable memory optimizations for the model"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Inizializza GradScaler per mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        running_train_loss = running_train_loss_1 = running_train_loss_2 = running_train_loss_3 = 0.0
        correct_train = total_train = 0
        perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)
        train_unstable_nodes = 0

        for index, (inputs, targets) in enumerate(self.train_data_loader):
            inputs, targets = inputs.to(device).to(torch.float32), targets.to(device)
            targets = targets.long()

            optimizer.zero_grad()
            outputs = model(inputs)

            # Calculate losses
            if isinstance(criterion, nn.MSELoss):
                targets_hot = F.one_hot(targets, num_classes=num_classes).float()
                loss = criterion(outputs, targets_hot)
            else:
                loss = criterion(outputs, targets)
            ce_loss = loss.item()

            # Backward pass con scaling
            with torch.cuda.amp.autocast():
                rsloss_conv, rs_loss_fc, unstable_nodes = self.get_rsloss(model=model, model_ref=model_ref, architecture_tuple=arch_tuple, input_batch=inputs, perturbation=perturbation, eps=eps)


            train_unstable_nodes += unstable_nodes
            total_loss = loss + CONV_RSLOSS_LAMBDA_MULTIPLIER * rsloss_lambda * rsloss_conv + rsloss_lambda * rs_loss_fc

            # Optimize
            scaler.scale(total_loss).backward()

            # Unscale gradients e clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            # Track metrics
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            running_train_loss += total_loss.item()
            running_train_loss_1 += ce_loss
            running_train_loss_2 += rsloss_conv.item()
            running_train_loss_3 += rs_loss_fc.item()

        # Calculate epoch statistics
        dataset_size = len(self.train_data_loader)
        epoch_stats = {
            'train_loss': running_train_loss / dataset_size,
            'train_accuracy': 100 * correct_train / total_train,
            'loss_1_train': running_train_loss_1 / dataset_size,
            'loss_2_train': running_train_loss_2,
            'loss_3_train': running_train_loss_3,
        }

        if DEBUG:
            self.logger.info("Epoch Statistics:")
            self.logger.info("  Train -> Loss: %.4f, Accuracy: %.2f%%" % 
                        (epoch_stats['train_loss'], epoch_stats['train_accuracy']))

        return train_unstable_nodes/len(self.train_data_loader)