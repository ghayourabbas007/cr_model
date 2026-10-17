"""PyTorch LSTM over the last `seq_len` days of features.

Two heads share the LSTM: a direction logit (BCE loss) and three ordered return quantiles
(pinball loss). Quantiles are parameterised as q10 + cumulative softplus increments, so they
can never cross. Standardisation statistics come from the training rows only.

The model needs the feature history preceding each row to build its input window; call
`set_history(features)` with the full (causal) feature frame before fit/predict.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from forecaster.features.build import feature_columns
from forecaster.modeling.base import QUANTILES, to_frame


def _torch():
    import torch
    return torch


class LSTMModel:
    name = "lstm"

    def __init__(self, horizon: int, seq_len: int = 30, hidden: int = 32, layers: int = 1,
                 dropout: float = 0.2, epochs: int = 60, lr: float = 1e-3, batch_size: int = 64,
                 patience: int = 8, val_frac: float = 0.15, weight_decay: float = 1e-4, seed: int = 42):
        self.horizon = horizon
        self.seq_len, self.hidden, self.layers, self.dropout = seq_len, hidden, layers, dropout
        self.epochs, self.lr, self.batch_size, self.patience = epochs, lr, batch_size, patience
        self.val_frac, self.weight_decay, self.seed = val_frac, weight_decay, seed
        self.history: pd.DataFrame | None = None

    # --- data -------------------------------------------------------------------------------
    def set_history(self, features: pd.DataFrame) -> LSTMModel:
        self.history = features
        return self

    def _windows(self, index: pd.Index) -> np.ndarray:
        if self.history is None:
            raise RuntimeError("Call set_history(features) before fit/predict")
        Z = ((self.history[self.features_] - self.mean_) / self.std_).clip(-5, 5).fillna(0.0)
        Z = Z.to_numpy(dtype=np.float32)
        pos = self.history.index.get_indexer(index)
        if (pos < 0).any():
            raise KeyError("Some rows are not present in the feature history")
        out = np.zeros((len(pos), self.seq_len, Z.shape[1]), dtype=np.float32)
        for i, p in enumerate(pos):
            lo = max(0, p - self.seq_len + 1)
            out[i, self.seq_len - (p - lo + 1):] = Z[lo : p + 1]  # left-pad with zeros (= mean)
        return out

    # --- network ----------------------------------------------------------------------------
    def _build(self, n_features: int):
        torch = _torch()
        nn = torch.nn
        hidden, layers, dropout = self.hidden, self.layers, self.dropout

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                                    dropout=dropout if layers > 1 else 0.0)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(hidden, 1 + len(QUANTILES))

            def forward(self, x):
                out, _ = self.lstm(x)
                z = self.head(self.drop(out[:, -1]))
                logit = z[:, 0]
                base, inc = z[:, 1:2], nn.functional.softplus(z[:, 2:])
                q = torch.cat([base, base + torch.cumsum(inc, dim=1)], dim=1)
                return logit, q

        return Net()

    @staticmethod
    def _loss(torch, logit, q, y_up, y_ret):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y_up)
        taus = torch.tensor(QUANTILES, dtype=q.dtype)
        err = y_ret.unsqueeze(1) - q
        pinball = torch.maximum(taus * err, (taus - 1) * err).mean()
        # Returns are ~0.1 in scale; weight pinball so both heads learn.
        return bce + 10.0 * pinball

    def fit(self, X: pd.DataFrame, y_up: pd.Series, y_ret: pd.Series) -> LSTMModel:
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.features_ = feature_columns(X)
        train_hist = self.history.loc[X.index, self.features_]
        self.mean_ = train_hist.mean()
        self.std_ = train_hist.std().replace(0, 1.0).fillna(1.0)
        self.mean_ = self.mean_.fillna(0.0)

        W = torch.tensor(self._windows(X.index))
        yu = torch.tensor(np.asarray(y_up, dtype=np.float32))
        yr = torch.tensor(np.asarray(y_ret, dtype=np.float32))

        n = len(W)
        n_val = int(n * self.val_frac)
        fit_end = n - n_val - self.horizon if n_val >= 30 else n  # purge between train and val
        net = self._build(W.shape[2])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad = np.inf, None, 0
        gen = torch.Generator().manual_seed(self.seed)

        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(fit_end, generator=gen)
            for b in range(0, fit_end, self.batch_size):
                idx = perm[b : b + self.batch_size]
                opt.zero_grad()
                logit, q = net(W[idx])
                loss = self._loss(torch, logit, q, yu[idx], yr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if fit_end == n:
                continue
            net.eval()
            with torch.no_grad():
                logit, q = net(W[n - n_val :])
                val = self._loss(torch, logit, q, yu[n - n_val :], yr[n - n_val :]).item()
            if val < best - 1e-5:
                best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net.eval()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        torch = _torch()
        with torch.no_grad():
            logit, q = self.net_(torch.tensor(self._windows(X.index)))
        return to_frame(X.index, torch.sigmoid(logit).numpy(), q.numpy())

    def __getstate__(self):
        # The network class is defined locally (lazy torch import), so persist weights only.
        # The (stale) feature history is not saved; call set_history() after loading.
        state = self.__dict__.copy()
        state["history"] = None
        if "net_" in state:
            state["net_"] = {k: v.cpu() for k, v in self.net_.state_dict().items()}
        return state

    def __setstate__(self, state):
        weights = state.get("net_")
        self.__dict__.update(state)
        if weights is not None:
            net = self._build(len(self.features_))
            net.load_state_dict(weights)
            self.net_ = net.eval()
