"""Small sparse value network designed for later integer Numba inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from nnue.features import INPUT_FEATURES, PADDING_INDEX


@dataclass(frozen=True, slots=True)
class NNUEConfig:
    feature_hidden: int = 128
    output_scale_cp: int = 400


class SparseNNUE(nn.Module):
    """Shared feature transformer with side-to-move ordered accumulators."""

    def __init__(self, config: NNUEConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = NNUEConfig()
        self.config = config
        self.feature = nn.Embedding(
            INPUT_FEATURES + 1,
            config.feature_hidden,
            padding_idx=PADDING_INDEX,
        )
        self.feature_bias = nn.Parameter(torch.zeros(config.feature_hidden))
        self.output = nn.Linear(config.feature_hidden, 1, bias=False)
        self.tempo = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.feature.weight[PADDING_INDEX].zero_()
        nn.init.zeros_(self.feature_bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.tempo)

    @staticmethod
    def clipped_relu(value: Tensor) -> Tensor:
        return torch.clamp(value, 0.0, 1.0)

    def forward(self, white: Tensor, black: Tensor, turn: Tensor) -> Tensor:
        white_accumulator = self.feature_bias + self.feature(white).sum(dim=1)
        black_accumulator = self.feature_bias + self.feature(black).sum(dim=1)
        white_to_move = turn.reshape(-1, 1) > 0
        us = torch.where(white_to_move, white_accumulator, black_accumulator)
        them = torch.where(white_to_move, black_accumulator, white_accumulator)
        us = self.clipped_relu(us)
        them = self.clipped_relu(them)
        raw = (self.output(us) - self.output(them)).squeeze(1) + self.tempo
        return raw * self.config.output_scale_cp


def save_checkpoint(
    path: str | Path,
    model: SparseNNUE,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "config": asdict(model.config),
        "model": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> SparseNNUE:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = SparseNNUE(NNUEConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device)
