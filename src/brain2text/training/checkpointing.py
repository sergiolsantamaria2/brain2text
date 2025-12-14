from __future__ import annotations
from pathlib import Path
import torch

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, payload: dict) -> str:
        path = self.dir / name
        torch.save(payload, path)
        return str(path)

    def load(self, path: str, map_location="cpu") -> dict:
        return torch.load(path, map_location=map_location, weights_only=False)
