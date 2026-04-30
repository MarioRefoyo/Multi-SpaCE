from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from scipy.sparse import (
    bsr_array,
    bsr_matrix,
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
)
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import Literal, get_args

from bin.utils.device import DEVICE

HANDLED_TYPES = Literal["np.ndarray", "scipy.sparse", "Tensor"]


def get_default_params() -> dict[str, Any]:
    return {
        "batch_size": 8,
        "patience": 200,
        "max_epochs": 1000,
    }


class NNClassifier(pl.LightningModule, ABC):
    def __init__(
        self, dropout: float = 0.2, lr: float = 1e-3, weight_decay: float = 0.1
    ):
        super(NNClassifier, self).__init__()
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def loss_function(self, batch: list[Tensor, Tensor]) -> Tensor:
        pass

    @abstractmethod
    def _forward(self, X: Tensor) -> Tensor:
        pass

    def forward(self, X: Any) -> Tensor:
        self.to(DEVICE)

        X = self._convert2tensor(X)

        return self._forward(X)

    def forward_cls(self, X: Tensor) -> Tensor:
        out = self.forward(X)
        out = torch.argmax(out, dim=1)
        return out

    def forward_np(self, X: Any) -> NDArray[np.float64]:
        out = self.forward(X)
        return out.detach().cpu().numpy()

    def forward_cls_np(self, X: Tensor) -> NDArray[np.int64]:
        out = self.forward_cls(X)
        return out.detach().cpu().numpy().astype(np.int64)

    def predict(self, X: Any) -> NDArray[np.float64]:
        return self.forward_np(X)

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any = None,
        y_test: Any = None,
        model_args: dict[str, Any] = get_default_params(),
    ) -> None:
        model_args = model_args | {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
        }

        X_train = self._convert2tensor(X_train)
        y_train = self._convert2tensor(y_train)
        if X_test is not None:
            X_test = self._convert2tensor(X_test)
            y_test = self._convert2tensor(y_test)

        train_dataloader = self._create_dataloader(
            X_train, y_train, **model_args
        )
        if X_test is not None:
            test_dataloader = self._create_dataloader(
                X_test, y_test, **model_args
            )
        else:
            test_dataloader = None

        trainer = self.get_trainer(**model_args)

        trainer.fit(
            model=self,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader,
        )

    def training_step(self, batch: list[Tensor, Tensor]) -> Tensor:
        loss = self.loss_function(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: list[Tensor, Tensor]) -> Tensor:
        loss = self.loss_function(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def get_trainer(self, **args: Any) -> pl.Trainer:
        callbacks: list[Callback] = []
        if args["X_test"] is not None:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=args["patience"],
                verbose=False,
                mode="min",
            )
            callbacks.append(early_stopping)

        model_checkpoint = ModelCheckpoint(
            filename="best_model-{epoch}-{val_loss:.4f}-{train_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
            every_n_epochs=1,
        )
        callbacks.append(model_checkpoint)

        return pl.Trainer(
            max_epochs=args["max_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            check_val_every_n_epoch=1,
            callbacks=callbacks,
            log_every_n_steps=10,
            enable_progress_bar=True,
        )

    def _convert2tensor(self, X: Any) -> Tensor:
        if isinstance(X, Tensor):
            pass
        elif isinstance(X, np.ndarray):
            X = Tensor(X.astype(np.float64))
        elif isinstance(
            X,
            (
                bsr_array,
                bsr_matrix,
                csc_array,
                csc_matrix,
                coo_array,
                coo_matrix,
                csr_array,
                csr_matrix,
            ),
        ):
            X = Tensor(X.toarray().astype(np.float64))
        else:
            raise ValueError(
                f"Unexpected type of the input. Only {get_args(HANDLED_TYPES)} are supported"
            )
        return X.to(DEVICE)

    def _create_dataloader(
        self, X: Tensor, y: Tensor, **args: Any
    ) -> DataLoader:
        dataset = TensorDataset(X, y)
        batch_size = int(args["batch_size"])
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader


class MLP(NNClassifier):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 32,
        n_layers: int = 3,
        dropout: float = 0.2,
        lr: float = 0.01,
        weight_decay: float = 0.1,
    ) -> None:
        super().__init__(dropout, lr, weight_decay)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.input_size = input_size
        self.output_size = output_size

        self._prepare_model()

    def loss_function(
        self, batch: list[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        X, y = batch
        y_hat = self.forward(X)
        loss = F.cross_entropy(y_hat, y, reduction="mean")
        return loss

    def _forward(self, X: Tensor) -> Tensor:
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def _prepare_model(self, **args: Any) -> None:
        layers: list[nn.Module] = []
        n_sizes = (
            [self.input_size]
            + [self.hidden_size] * self.n_layers
            + [self.output_size]
        )

        for idx in range(len(n_sizes) - 2):
            layers.append(nn.Linear(n_sizes[idx], n_sizes[idx + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(n_sizes[-2], n_sizes[-1]))
        layers.append(nn.Softmax())

        self.layers = nn.Sequential(*layers)


class LSTM(NNClassifier):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
    ):
        super().__init__(dropout, lr, weight_decay)
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.n_layers = num_layers

        self.input_size = input_size
        self.output_size = output_size

        self._prepare_model()

    def loss_function(
        self, batch: list[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        X, y = batch
        y_hat = self.forward(X)

        loss = F.cross_entropy(y_hat, y.long(), reduction="mean")
        return loss

    def _forward(self, X: Tensor) -> Tensor:

        h0 = torch.zeros(self.n_layers, X.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.n_layers, X.size(0), self.hidden_size).to(DEVICE)

        out, _ = self.lstm(X, (h0, c0))
        out = out[:, -1, :]

        out = self.fc(out)
        out = self.softmax(out)
        return out

    def _prepare_model(self) -> None:

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.n_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax()
