import numpy as np
import pandas as pd
import torch
from torch import nn

from .base import PremiumModel
from .nn import MLP, gamma_loss, make_train_val_multdataloaders, poisson_loss, train_pytorch_model


class MTMoENN(nn.Module):
    def __init__(self, shared_encoder_params, freq_decoder_params, sev_decoders_params):
        super().__init__()
        self.shared_encoder = MLP(**shared_encoder_params)  # type: ignore
        self.freq_decoder = MLP(**freq_decoder_params, output_dim=1, final_activation=nn.Softplus())  # type: ignore
        self.sev_decoders = nn.ModuleList(
            MLP(
                input_dim=shared_encoder_params["hidden_dims"][-1],
                hidden_dims=hidden_dims,
                dropout_rate=sev_decoders_params["dropout_rate"],
                output_dim=2,
                final_activation=nn.Softplus(),
            )
            for hidden_dims in sev_decoders_params["hidden_dims"]
        )

    @property
    def n_experts(self):
        return len(self.sev_decoders)

    def forward(self, X):
        if isinstance(X, tuple):
            return tuple(self.forward(x) for x in X)

        shared_features = self.shared_encoder(X)
        freq_out = self.freq_decoder(shared_features)
        sev_weights = self.compute_expert_weights(freq_out.squeeze())
        sev_out = torch.stack([sev_decoder(shared_features) for sev_decoder in self.sev_decoders], dim=1)
        sev_out = torch.sum(sev_weights.unsqueeze(2) * sev_out, dim=1)
        return freq_out, sev_out

    @torch.no_grad()
    def compute_expert_weights(self, lambda_pred):
        # Compute Poisson probabilities for frequencies 0, 1, 2, ..., m+
        weights = torch.softmax(torch.distributions.Poisson(lambda_pred[:, None]).log_prob(1 + torch.arange(self.n_experts, device=lambda_pred.device)), dim=1)
        return weights


class MTMoENNPremiumModel(PremiumModel):
    cat_encode_mode = "dummies"
    fit_scaler = True

    model: nn.Module
    device: torch.device

    def __init__(
        self,
        shared_encoder_params={
            "hidden_dims": [30, 15],
            "activation": nn.ReLU(),
            "dropout_rate": 0.5,
        },
        freq_decoder_params={
            "hidden_dims": [24, 12],
            "activation": nn.ReLU(),
            "dropout_rate": 0.5,
        },
        sev_decoders_params={
            "hidden_dims": [(20, 10), (18, 9), (18, 9), (18, 9), (10, 5)],
            "activation": nn.ReLU(),
            "dropout_rate": 0.5,
        },
        training_params={
            "batch_size": 128,
            "learning_rate": 0.0001,
            "weight_decay": 0.0008,
            "patience": 10,
            "validation_split": 0.2,
            "num_epochs": 1000,
            "verbose": True,
            "log_interval": 20,
            "seed": 42,
            "freq_weight": 2.0,
            "sev_weight": 0.2,
        },
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.shared_encoder_params = shared_encoder_params
        self.freq_decoder_params = freq_decoder_params
        self.sev_decoders_params = sev_decoders_params
        self.training_params = training_params
        self.device = device

    @property
    def n_experts(self):
        return len(self.sev_decoders_params["hidden_dims"])

    def _fit(self, X_freq, y_freq, w_freq, X_sev, y_sev, w_sev):
        sev_mask = np.isin(X_freq.index, X_sev.index)
        X_sev = X_freq[sev_mask]  # get full features

        # Create models with proper input dimensions
        hidden_dim = self.shared_encoder_params["hidden_dims"][-1]
        self.shared_encoder_params["input_dim"] = X_freq.shape[1]
        self.freq_decoder_params["input_dim"] = hidden_dim
        self.sev_decoders_params["input_dim"] = hidden_dim
        self.model = MTMoENN(self.shared_encoder_params, self.freq_decoder_params, self.sev_decoders_params).to(self.device)
        if self.training_params["verbose"]:
            print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

        train_loader, val_loader = make_train_val_multdataloaders(
            (X_freq, y_freq, w_freq),
            (X_sev, y_sev, w_sev),
            device=self.device,
            batch_size=self.training_params["batch_size"],
            val_size=self.training_params["validation_split"],
            random_state=self.training_params["seed"],
        )

        def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor):
            freq_pred, sev_pred = y_pred[0][0], y_pred[1][1]
            freq_true, sev_true = y_true
            freq_weights, sev_weights = weights
            freq_loss = poisson_loss(y_pred=freq_pred, y_true=freq_true, weights=freq_weights)
            sev_loss = gamma_loss(y_pred=sev_pred, y_true=sev_true, weights=sev_weights)
            return self.training_params["freq_weight"] * freq_loss + self.training_params["sev_weight"] * sev_loss

        train_pytorch_model(
            model=self.model,
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

    def _predict(self, X_freq, X_sev):
        self.model.eval()

        X = torch.tensor(X_freq.values, device=self.device, dtype=torch.float32)

        with torch.inference_mode():
            freq_out, sev_out = self.model(X)
            lambda_ = freq_out[:, 0]
            alpha = sev_out[:, 0]
            beta = sev_out[:, 1]
            freq_pred = lambda_.cpu().numpy()
            sev_pred = (alpha * beta).cpu().numpy()

        return pd.Series(freq_pred, index=X_freq.index, name="frequency"), pd.Series(sev_pred, index=X_sev.index, name="severity")

    def _save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "scalers": self.scalers,
                "shared_encoder_params": self.shared_encoder_params,
                "freq_decoder_params": self.freq_decoder_params,
                "sev_decoders_params": self.sev_decoders_params,
                "training_params": self.training_params,
                "device": self.device,
            },
            path,
        )

    @classmethod
    def _load(cls, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self = cls(
            shared_encoder_params=checkpoint["shared_encoder_params"],
            freq_decoder_params=checkpoint["freq_decoder_params"],
            sev_decoders_params=checkpoint["sev_decoders_params"],
            training_params=checkpoint["training_params"],
            device=checkpoint["device"],
        )
        self.model = MTMoENN(self.shared_encoder_params, self.freq_decoder_params, self.sev_decoders_params).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.scalers = checkpoint["scalers"]
        return self
