import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int | None = None,
        dropout_rate: float = 0.0,
        activation=nn.ReLU(),
        final_activation=None,
    ):
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), activation, nn.Dropout(dropout_rate)]
            prev_dim = h
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
        if final_activation is not None:
            if not isinstance(final_activation, nn.Module):
                class CustomActivation(nn.Module):
                    def forward(self, x):
                        return final_activation(x)
                layers.append(CustomActivation())
            else:
                layers.append(final_activation)
        super().__init__(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class MultiDataLoader:
    def __init__(self, *datasets: TensorDataset, batch_size: int = 128, shuffle: bool = True) -> None:
        self.loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) for dataset in datasets]
        self.max_length = max(len(loader) for loader in self.loaders)
        self.dataset = max(datasets, key=len)

    def __iter__(self):
        self.iterators = [iter(loader) for loader in self.loaders]
        self.count = 0
        return self

    def __next__(self):
        if self.count >= self.max_length:
            raise StopIteration

        result = []
        for i, it in enumerate(self.iterators):
            try:
                batch = next(it)
            except StopIteration:
                # Reset the iterator if exhausted
                self.iterators[i] = iter(self.loaders[i])
                batch = next(self.iterators[i])
            result.append(batch)

        self.count += 1
        return zip(*result)


def mean_possion_loglikelihood(*, lambda_: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor):
    return torch.distributions.Poisson(lambda_ * weights, validate_args=False).log_prob(y_true).mean()

def mean_gamma_loglikelihood(*, alpha: torch.Tensor, theta: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor):
    return torch.distributions.Gamma(concentration=weights * alpha, rate=weights / theta).log_prob(y_true).mean()

def poisson_loss(*, y_pred, y_true, weights):
    assert y_pred.shape[-1] == 1, "y_pred must have shape (..., 1) for lambda"
    return -mean_possion_loglikelihood(lambda_=y_pred[..., 0], y_true=y_true, weights=weights)

def gamma_loss(*, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor):
    assert y_pred.shape[-1] == 2, "y_pred must have shape (..., 2) for alpha and theta"
    return -mean_gamma_loglikelihood(alpha=y_pred[..., 0], theta=y_pred[..., 1], y_true=y_true, weights=weights)


def train_pytorch_model(model, train_loader, val_loader, loss_fn, num_epochs=100, learning_rate=0.001, weight_decay=1e-4, patience=10, log_interval=20, verbose=True, seed=42):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)

    # Initialize parameters
    for param in model.parameters():
        if param.requires_grad:
            match param.dim():
                case 0:
                    pass # should be initialized
                case 1:
                    nn.init.zeros_(param)
                case 2:
                    nn.init.xavier_uniform_(param)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = sum(_train_step(model, optimizer, loss_fn, batch) for batch in train_loader)
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = sum(_validate_step(model, loss_fn, batch) for batch in val_loader)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

        if verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


def _train_step(model, optimizer, loss_fn, batch):
    X_batch, y_batch, w_batch = batch

    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = loss_fn(y_pred=y_pred, y_true=y_batch, weights=w_batch)

    assert not torch.isnan(loss) and not torch.isinf(loss)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item() * len(X_batch)


def _validate_step(model, loss_fn, batch):
    X_batch, y_batch, w_batch = batch

    y_pred = model(X_batch)
    loss = loss_fn(y_pred=y_pred, y_true=y_batch, weights=w_batch)

    assert not torch.isnan(loss) and not torch.isinf(loss)

    return loss.item() * len(X_batch)


def as_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, (list, tuple, np.ndarray)):
        return torch.tensor(data, dtype=torch.float32)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return torch.tensor(data.values, dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor, list, tuple, np.ndarray, or pd.DataFrame.")


def make_train_val_datasets(*data, device, val_size=0.2, random_state=None) -> tuple[TensorDataset, TensorDataset]:
    if random_state is not None:
        torch.manual_seed(random_state)

    tensors = [as_tensor(d).to(device) for d in data]
    n_samples = len(tensors[0])
    indices = torch.randperm(n_samples)
    val_size = int(n_samples * val_size)
    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]
    return TensorDataset(*(t[train_idx] for t in tensors)), TensorDataset(*(t[val_idx] for t in tensors))


def make_train_val_dataloaders(arrs: tuple, device, batch_size=128, val_size=0.2, random_state=None) -> tuple[DataLoader, DataLoader]:
    train_loader, val_loader = (DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in make_train_val_datasets(*arrs, device=device, val_size=val_size, random_state=random_state))
    return train_loader, val_loader


def make_train_val_multdataloaders(*arrs: tuple, device, batch_size=128, val_size=0.2, random_state=None) -> tuple[MultiDataLoader, MultiDataLoader]:
    train_loader, val_loader = (MultiDataLoader(freq_set, sev_set, batch_size=batch_size, shuffle=True) for freq_set, sev_set in zip(*(make_train_val_datasets(*a, device=device, val_size=val_size, random_state=random_state) for a in arrs)))
    return train_loader, val_loader
