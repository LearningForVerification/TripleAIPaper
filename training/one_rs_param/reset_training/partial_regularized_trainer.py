import copy
import logging
import time
from abc import ABC, abstractmethod
from configparser import ConfigParser
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import PerturbationLpNorm, BoundedModule
from torch import Tensor, optim
from torch.utils.data import DataLoader
from training.utils.rs_loss_regularizer import  calculate_rs_loss_regularizer_resnet18



from training.one_rs_param import device


class ModelTrainingManager(ABC):
    def __init__(
        self,
        data_loader: Tuple[DataLoader, DataLoader],
        config: ConfigParser,
        verbose: bool = True,
        init_model_path: str | None = None
    ):
        self.logger = logging.getLogger(__name__)
        self.train_data_loader, self.test_loader = data_loader
        self.config = config
        self.device = device
        self.verbose = verbose
        self.init_model_path = init_model_path

    def get_rsloss(
            self,
            model: nn.Module,
            model_ref: nn.Module,
            architecture_tuple: tuple,
            input_batch: Tensor,
            perturbation,
            eps: float,
            method: str = "ibp"
    ) -> Tuple[Tensor, int]:
        """
        Calcola RS loss e numero di neuroni instabili su una frazione del batch (10%)
        """

        _time = time.time()

        batch_size = input_batch.shape[0]
        rs_batch_size = max(1, int(0.1 * batch_size))  # almeno 1 sample

        # 🔹 campionamento casuale degli indici
        idx = torch.randperm(batch_size, device=input_batch.device)[:rs_batch_size]
        input_rs = input_batch[idx]

        # 🔹 forward backbone SOLO su sotto-batch
        with torch.no_grad():
            backbone_output = model_ref.forward_backbone(input_rs)

        # 🔹 intervalli
        lb = backbone_output - eps
        ub = backbone_output + eps

        # 🔹 RS loss
        rs_loss, n_unstable_nodes = calculate_rs_loss_regularizer_resnet18(
            model_ref,
            lb,
            ub,
            normalized=True
        )

        diff = time.time() - _time
        print("rs_loss_time (10% batch):", diff)

        return rs_loss, n_unstable_nodes

    # ==========================================================
    # TRAIN
    # ==========================================================
    def train(
        self,
        model_untrained: nn.Module,
        arch_tuple: tuple,
        dummy_input: Tensor,
        data_dict: Dict[str, Any],
        num_epochs: int,
        rsloss_lambda: float,
        eps: float | None = None,
    ):
        # -------- Clone & wrap model --------
        model_ref = copy.deepcopy(model_untrained).to(self.device)

        # -------- Carica backbone opzionale --------
        if self.init_model_path is not None:
            print(f"Carico pesi backbone da: {self.init_model_path}")
            backbone_state = torch.load(self.init_model_path)
            model_state = model_ref.state_dict()
            for k in backbone_state:
                if "fc1" not in k and "fc2" not in k and k in model_state:
                    model_state[k] = backbone_state[k]
            model_ref.load_state_dict(model_state)

            # Congela parametri del backbone
            for name, param in model_ref.named_parameters():
                if "fc1" not in name and "fc2" not in name:
                    param.requires_grad = False
            print("Backbone caricato e congelato. FC1 e FC2 addestrabili.")

        model = BoundedModule(model_ref, dummy_input, device=self.device)

        # -------- Optimizer (solo parametri addestrabili) --------
        opt_cfg = data_dict["optimizer"].copy()
        opt_type = opt_cfg.pop("type")
        optimizer_cls = {
            "Adam": optim.Adam,
            "SGD": optim.SGD
        }[opt_type]

        optimizer = optimizer_cls(filter(lambda p: p.requires_grad, model.parameters()), **opt_cfg)

        # -------- Loss --------
        loss_name = data_dict["training"]["loss_name"]
        num_classes = int(data_dict["data"]["output_dim"])
        criterion = {
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "MSE": nn.MSELoss
        }[loss_name]()

        # -------- LR scheduler --------
        if self.config.getboolean("fixed_lr"):
            lr_lambda = lambda epoch: 1.0
        else:
            decay = self.config.getfloat("lr_decay")
            cycle = self.config.getint("lambda_lr_cycle")
            lr_lambda = lambda epoch: decay ** (epoch // cycle)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ======================================================
        # TRAINING LOOP
        # ======================================================
        for epoch in range(num_epochs):
            start = time.time()

            train_unstable = self._train_epoch(
                model=model,
                model_ref=model_ref,
                optimizer=optimizer,
                criterion=criterion,
                arch_tuple=arch_tuple,
                num_classes=num_classes,
                rsloss_lambda=rsloss_lambda,
                eps=eps,
                scheduler=scheduler
            )

            elapsed = time.time() - start
            self.logger.debug(f"Epoch {epoch+1}/{num_epochs} - {elapsed:.2f}s")

            # Validazione periodica
            if epoch > 0 and epoch % self.config.getint("validation_frequency") == 0:
                self.calculate_accuracy_and_loss(
                    model, model_ref, arch_tuple,
                    criterion, num_classes,
                    rsloss_lambda, train_set=False, eps=eps
                )

        # -------- Final evaluation --------
        test_stats = self.calculate_accuracy_and_loss(
            model, model_ref, arch_tuple,
            criterion, num_classes,
            rsloss_lambda, train_set=False, eps=eps
        )

        train_stats = self.calculate_accuracy_and_loss(
            model, model_ref, arch_tuple,
            criterion, num_classes,
            rsloss_lambda, train_set=True, eps=eps
        )

        # -------- Score finale --------
        score = {
            "train_accuracy": train_stats[0],
            "test_accuracy": test_stats[0],
            "train_loss": train_stats[1],
            "test_loss": test_stats[1],
            "rs_train_loss": train_stats[3],
            "rs_test_loss": test_stats[3],
            "train_unstable_nodes": train_stats[4],
            "test_unstable_nodes": test_stats[4],
            "lambda": rsloss_lambda,
            "eps": eps,
            "architecture": arch_tuple
        }

        return score, model, model_ref

    # ==========================================================
    # SINGLE EPOCH TRAINING
    # ==========================================================
    def _train_epoch(
        self,
        model,
        model_ref,
        optimizer,
        criterion,
        arch_tuple,
        num_classes,
        rsloss_lambda,
        eps,
        scheduler=None,
        n_points_bounds: int = 5,
        num_samples_mc: int = 20
    ):
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)

        unstable_nodes_epoch = 0
        correct = total = 0
        running_loss = 0.0

        for inputs, targets in self.train_data_loader:
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device).long()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Forward standard
                outputs = model(inputs)

                # CE loss
                if isinstance(criterion, nn.MSELoss):
                    targets_oh = F.one_hot(targets, num_classes).float()
                    ce_loss = criterion(outputs, targets_oh)
                else:
                    ce_loss = criterion(outputs, targets)

                # ===============================
                # RS LOSS su FC1 e FC2
                # ===============================
                rs_loss = torch.tensor(0.0, device=self.device)
                if rsloss_lambda > 0:
                    # Seleziona n_points casuali
                    rs_loss, unstable_nodes_epoch = self.get_rsloss(
                        model_ref=model_ref, eps= eps, model= model, input_batch=inputs, perturbation=None, architecture_tuple=arch_tuple)



                # Loss totale
                total_loss = ce_loss + rsloss_lambda * rs_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            _, pred = torch.max(outputs, 1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            running_loss += total_loss.item()

        return unstable_nodes_epoch / len(self.train_data_loader)

    # ==========================================================
    # ACCURACY + LOSS
    # ==========================================================
    def calculate_accuracy_and_loss(
        self,
        model,
        model_ref,
        arch_tuple,
        criterion,
        num_classes,
        rsloss_lambda,
        train_set,
        eps
    ):
        model.eval()
        loader = self.train_data_loader if train_set else self.test_loader
        perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)

        total_loss = ce_loss_tot = rs_loss_tot = 0.0
        unstable_nodes = 0
        correct = total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).long()

                outputs = model(inputs)

                if isinstance(criterion, nn.MSELoss):
                    targets_oh = F.one_hot(targets, num_classes).float()
                    ce_loss = criterion(outputs, targets_oh)
                else:
                    ce_loss = criterion(outputs, targets)

                rs_loss = torch.tensor(0.0, device=self.device)
                if rsloss_lambda > 0:
                    rs_loss, unstable = self.get_rsloss(
                        model, model_ref, arch_tuple,
                        inputs,
                        perturbation, eps
                    )
                    unstable_nodes += unstable

                total_loss += ce_loss.item() + rsloss_lambda * rs_loss.item()
                ce_loss_tot += ce_loss.item()
                rs_loss_tot += rs_loss.item()

                _, pred = torch.max(outputs, 1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)

        accuracy = 100 * correct / total if total > 0 else 0
        n = len(loader)

        return (
            accuracy,
            total_loss / n,
            ce_loss_tot / n,
            rs_loss_tot / n,
            unstable_nodes / n
        )

    # ==========================================================
    # FUNZIONE DI STIMA BOUNDS DELLA BACKBONE (opzionale)
    # ==========================================================
    def compute_backbone_bounds_during_training(
        self,
        model: nn.Module,
        eps: float,
        n_points: int = 10,
        num_samples: int = 50
    ):
        """
        Stima lower e upper bounds della backbone durante il training.
        Non influisce sulla loss.
        """
        model.eval()
        model.to(self.device)

        inputs, _ = next(iter(self.train_data_loader))
        inputs = inputs.to(self.device).float()

        n_points = min(n_points, inputs.size(0))
        idx = torch.randperm(inputs.size(0))[:n_points]
        x0_batch = inputs[idx]

        all_outputs = []
        with torch.no_grad():
            for x0 in x0_batch:
                x0 = x0.unsqueeze(0)
                samples = x0 + (torch.rand(num_samples, *x0.shape[1:], device=self.device) * 2 - 1) * eps
                samples = samples.clamp(0.0, 1.0)
                outputs = model.forward_backbone(samples)
                all_outputs.append(outputs)

        all_outputs = torch.cat(all_outputs, dim=0)
        lower_bounds = torch.min(all_outputs, dim=0)[0]
        upper_bounds = torch.max(all_outputs, dim=0)[0]

        return lower_bounds, upper_bounds
