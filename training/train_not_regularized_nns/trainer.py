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

from training.train_not_regularized_nns import VALIDATION_FREQUENCY, ACCURACY_THRESHOLD, LAMBDA_LR_CYCLE, FIXED_LR, LR_DECAY, DEBUG

class ModelTrainingManager:
    def __init__(self, target_acc: float,
                 data_loader: Tuple[DataLoader, DataLoader],
                 verbose: bool = True
                 ):
        self.scheduler_type_priority = [
            'Constant',
            'LambdaLR',
            'StepLR',
            'MultiStepLR',
            'CosineAnnealingLR'
        ]
        self.logger = logging.getLogger(__name__)
        self.target_accuracy = target_acc
        self.train_data_loader, self.test_loader = data_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.verbose = verbose

    def find_overfitted_best_model(self, model_untr: nn.Module, arch_tuple: tuple, data_dict: Dict[str, Any],
                                   max_num_epochs: int, early_stopping: bool = False, previous_model: nn.Module = None):
        """
        Trains multiple times with different schedulers until a model reaches the target accuracy.
        If a previous_model is provided, reuse overlapping weights where possible.
        """
        model_untr = copy.deepcopy(model_untr)

        if previous_model is not None:
            model_untr = self._reuse_weights(model_untr, previous_model)

        success_flag = False

        while not success_flag and self.scheduler_type_priority:
            scheduler_type = self.scheduler_type_priority.pop(0)
            score, model = self.train(
                model_untr,
                arch_tuple,
                data_dict,
                max_num_epochs,
                scheduler_type,
                early_stopping,
                previous_model
            )

            if score['test_accuracy'] > self.target_accuracy:
                # Return the model and scores if accuracy thresholds are met
                return score, model
            else:
                self.logger.info(
                    f"Failed (acc: {score['test_accuracy']:.4f}). Target acc: {self.target_accuracy} Trained with scheduler: {scheduler_type}")

        return None, None

    def train(self,
        model_untr: nn.Module,
        arch_tuple: tuple,
        data_dict: Dict[str, Any],
        num_epochs: int,
        scheduler_type: str,
        early_stopping: bool = False,
        previous_model: nn.Module = None,

        ):

        model = copy.deepcopy(model_untr)
        model.to(device=self.device)

        if previous_model is not None:
            # Qui va la logica per previous_model (se necessaria)
            pass

        """Same docstring but updated for single training mode"""
        # Validate required data_dict structure
        required_keys = ['optimizer', 'scheduler_lr', 'data', 'training']
        if not all(key in data_dict for key in required_keys):
            raise ValueError(f"data_dict missing required keys: {required_keys}")

        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

        # Unpack data_dict
        optimizer_dict = data_dict['optimizer']

        # Create optimizer params dict
        opt_params = optimizer_dict.copy()
        optimizer_name = opt_params.pop('type')

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

        optimizer = optimizer_cls(model.parameters(), **opt_params)
        criterion = criterion_cls()

        if scheduler_type == 'Constant':
            LR_DECAY = 0.99
            LAMBDA_LR_CYCLE = 400
            lambda_lr = lambda epoch: 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

        elif scheduler_type == 'LambdaLR':
            LR_DECAY = 0.99
            LAMBDA_LR_CYCLE = 400
            lambda_lr = lambda epoch: LR_DECAY ** (epoch // LAMBDA_LR_CYCLE)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

        elif scheduler_type == 'StepLR':
            step_size = 50
            gamma = 0.8
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


        elif scheduler_type == 'MultiStepLR':
            milestones = [30, 60, 90]
            gamma = 0.1
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        elif scheduler_type == 'CosineAnnealingLR':
            T_max = 20  # numero di epoche per ciclo completo
            eta_min = 1e-5  # minimo learning rate
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        perfect_train_acc_epochs = 0

        # Training the model
        for epoch in range(num_epochs):
            self.logger.debug("Epoch %d/%d" % (epoch + 1, num_epochs))
            train_loss, train_acc  = self._train_epoch_and_get_stats(model=model, device=self.device,
                                            optimizer=optimizer, criterion=criterion, num_classes=num_classes,
                                            scheduler=scheduler, early_stopping = early_stopping)

            if early_stopping:
                if train_acc == 100.0:
                    perfect_train_acc_epochs += 1
                else:
                    perfect_train_acc_epochs = 0

            if epoch % VALIDATION_FREQUENCY == 0 and epoch != 0:
                self.logger.debug("Evaluating model on test set at epoch %d/%d" % (epoch + 1, num_epochs))
                test_accuracy, test_loss = self.calculate_accuracy_and_loss(
                    model, criterion, num_classes,
                    train_set=False)

            if early_stopping and perfect_train_acc_epochs >= 20:
                test_accuracy, test_loss = self.calculate_accuracy_and_loss(
                    model, criterion, num_classes,
                    train_set=False)
                if test_accuracy >= self.target_accuracy + ACCURACY_THRESHOLD:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}: 100%% train accuracy maintained for 20 epochs with better accuracy than target.")
                    break

        # Calculate final metrics
        test_accuracy, test_loss = self.calculate_accuracy_and_loss(
            model, criterion, num_classes, train_set=False)
        train_accuracy, train_loss = self.calculate_accuracy_and_loss(
            model, criterion, num_classes, train_set=True)

        score = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'architecture': arch_tuple,
        }

        self.logger.info("Training completed with following metrics:")
        self.logger.info("Train accuracy: %.2f%%" % train_accuracy)
        self.logger.info("Test accuracy: %.2f%%" % test_accuracy)
        self.logger.info("Train loss: %.4f" % train_loss)
        self.logger.info("Test loss: %.4f" % test_loss)

        return score, model

    def _calculate_loss(
            self,
            model: nn.Module,
            loss_criterion: nn.Module,
            num_classes: int,
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

        data_loader = self.train_data_loader if train_set else self.test_loader
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device).to(torch.float32), targets.to(self.device)
                targets = targets.long()  # correggo il missing assignment

                outputs = model(inputs)

                if isinstance(loss_criterion, nn.MSELoss):
                    targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                    loss = loss_criterion(outputs, targets_hot_encoded)
                else:
                    loss = loss_criterion(outputs, targets)

                running_loss += loss.item()

                # Pulizia memoria
                del outputs
                del loss
                if 'targets_hot_encoded' in locals():
                    del targets_hot_encoded
                torch.cuda.empty_cache()

        average_loss = running_loss / len(data_loader)
        return average_loss

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
            loss_criterion: nn.Module,
            num_classes: int,
            train_set: bool = False) -> tuple:
        """Calculate accuracy and loss for either training or test dataset"""

        total_loss = self._calculate_loss(model, loss_criterion, num_classes, train_set=train_set)
        accuracy = self._calculate_accuracy(model, train_set)

        if hasattr(self, 'verbose') and self.verbose:
            set_name = "Training" if train_set else "Test"
            self.logger.info("Statistics on %s Set:" % set_name)
            self.logger.info("  Loss: %.4f, Accuracy: %.2f%%" % (total_loss, accuracy))

        return accuracy, total_loss

    def _train_epoch_and_get_stats(self, model, device, optimizer, criterion, num_classes,
                                   scheduler=None, early_stopping = False):
        model.train()

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scaler = torch.cuda.amp.GradScaler()

        running_train_loss = running_train_loss_1 = 0.0
        correct_train = total_train = 0

        for index, (inputs, targets) in enumerate(self.train_data_loader):
            inputs, targets = inputs.to(device).to(torch.float32), targets.to(device)
            targets = targets.long()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)

                if isinstance(criterion, nn.MSELoss):
                    targets_hot = F.one_hot(targets, num_classes=num_classes).float()
                    loss = criterion(outputs, targets_hot)
                else:
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            total_train += targets.size(0)

            if isinstance(criterion, nn.CrossEntropyLoss):
                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == targets).sum().item()

            if index % 10 == 0 and self.verbose:
                self.logger.info(f"Batch {index}/{len(self.train_data_loader)} Loss: {loss.item():.4f}")

        if scheduler is not None:
            scheduler.step()

        train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0

        if self.verbose:
            self.logger.info(f"Train Loss: {running_train_loss / len(self.train_data_loader):.4f}")
            self.logger.info(f"Train Accuracy: {train_accuracy:.2f}%")

        return running_train_loss / len(self.train_data_loader), train_accuracy



    def _reuse_weights(self, new_model: nn.Module, old_model: nn.Module) -> nn.Module:
        """
        Return a new model with weights reused from old_model wherever possible.
        Assumes both models have the same architecture structure (e.g., only Linear layers + ReLU),
        and that new_model has larger dimensions in the Linear layers.
        """
        # Deep copy to avoid modifying original new_model
        reused_model = copy.deepcopy(new_model)

        # Get state_dicts
        new_state = reused_model.state_dict()
        old_state = old_model.state_dict()

        if DEBUG:
            for name in new_state:
                print(f"name new model: {name}")

            for name in old_state:
                print(f"name old model: {name}")


        for name in new_state:
            if name in old_state:
                old_param = old_state[name]
                new_param = new_state[name]

                try:
                    if old_param.ndim == 2:  # weights (out_features, in_features)
                        if DEBUG:
                            print(f"old params: {old_param.shape}")
                            print(f"new params: {new_param.shape}")

                        out_dim, in_dim = old_param.shape
                        new_state[name][:out_dim, :in_dim] = old_param
                        self.logger.info(f"Copied Linear weights for {name}")
                    elif old_param.ndim == 1:  # bias (out_features,)
                        out_dim = old_param.shape[0]
                        new_state[name][:out_dim] = old_param
                        self.logger.info(f"Copied Linear bias for {name}")
                except Exception as e:
                    self.logger.warning(f"Could not copy weights for {name} due to shape mismatch: {e}")
            else:
                self.logger.warning(f"{name} not found in old_model")

        # Load updated state_dict
        reused_model.load_state_dict(new_state)
        return reused_model
