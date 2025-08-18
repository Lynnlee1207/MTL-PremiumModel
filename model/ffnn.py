import pandas as pd
import torch
from torch import nn

from .base import PremiumModel
from .nn import MLP, gamma_loss, make_train_val_dataloaders, poisson_loss, train_pytorch_model


class FFNN(nn.Module):
    def __init__(self, frequency_params, severity_params):
        super().__init__()
        self.frequency_model = MLP(**frequency_params, final_activation=nn.Softplus(), output_dim=1)  # type: ignore
        self.severity_model = MLP(**severity_params, final_activation=nn.Softplus(), output_dim=2)  # type: ignore

    def forward(self, X_freq, X_sev):
        freq_out = self.frequency_model(X_freq)
        sev_out = self.severity_model(X_sev)
        return freq_out, sev_out


class FFNNPremiumModel(PremiumModel):
    cat_encode_mode = "dummies"
    fit_scaler = True

    model: nn.Module
    device: torch.device

    def __init__(
        self,
        frequency_params={
            "hidden_dims": [50, 25],
            "activation": nn.ReLU(),
            "dropout_rate": 0.2,
        },
        severity_params={
            "hidden_dims": [50, 25],
            "activation": nn.ReLU(),
            "dropout_rate": 0.2,
        },
        training_params={
            "batch_size": 128,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 15,
            "validation_split": 0.2,
            "num_epochs": 1000,
            "verbose": True,
            "log_interval": 20,
            "seed": 42,
        },
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.frequency_params = frequency_params
        self.severity_params = severity_params
        self.training_params = training_params
        self.device = device

    def _fit(self, X_freq, y_freq, w_freq, X_sev, y_sev, w_sev):
        # Create models with proper input dimensions
        self.frequency_params["input_dim"] = X_freq.shape[1]
        self.severity_params["input_dim"] = X_sev.shape[1]
        self.model = FFNN(self.frequency_params, self.severity_params).to(self.device)
        if self.training_params["verbose"]:
            print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

        if self.training_params["verbose"]:
            print("Training frequency model...")
        self._train(X_freq, y_freq, w_freq, self.model.frequency_model, poisson_loss)

        if self.training_params["verbose"]:
            print("Training severity model...")
        self._train(X_sev, y_sev, w_sev, self.model.severity_model, gamma_loss)

    def _predict(self, X_freq, X_sev):
        self.model.eval()

        X_freq_tensor = torch.tensor(X_freq.values, device=self.device, dtype=torch.float32)
        X_sev_tensor = torch.tensor(X_sev.values, device=self.device, dtype=torch.float32)

        with torch.inference_mode():
            freq_out, sev_out = self.model(X_freq_tensor, X_sev_tensor)
            lambda_ = freq_out[:, 0]
            alpha = sev_out[:, 0]
            beta = sev_out[:, 1]
            freq_pred = lambda_.cpu().numpy()
            sev_pred = (alpha * beta).cpu().numpy()

        return pd.Series(freq_pred, index=X_freq.index, name="frequency"), pd.Series(sev_pred, index=X_sev.index, name="severity")

    def _train(self, X, y, w, model, loss_fn):
        assert (w > 0).all().item()

        train_loader, val_loader = make_train_val_dataloaders(
            (X, y, w),
            device=self.device,
            batch_size=self.training_params["batch_size"],
            val_size=self.training_params["validation_split"],
            random_state=self.training_params["seed"],
        )

        train_pytorch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            num_epochs=self.training_params["num_epochs"],
            learning_rate=self.training_params["learning_rate"],
            patience=self.training_params["patience"],
            verbose=self.training_params["verbose"],
            log_interval=self.training_params["log_interval"],
            weight_decay=self.training_params["weight_decay"],
            seed=self.training_params["seed"],
        )

    def _save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "scalers": self.scalers,
                "frequency_params": self.frequency_params,
                "severity_params": self.severity_params,
                "training_params": self.training_params,
                "device": self.device,
            },
            path,
        )

    @classmethod
    def _load(cls, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self = cls(
            frequency_params=checkpoint["frequency_params"],
            severity_params=checkpoint["severity_params"],
            training_params=checkpoint["training_params"],
            device=checkpoint["device"],
        )
        self.model = FFNN(self.frequency_params, self.severity_params).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.scalers = checkpoint["scalers"]
        return self
