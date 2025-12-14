from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import subprocess

@dataclass
class RunPaths:
    run_dir: Path
    output_dir: Path
    checkpoint_dir: Path

class RunManager:
    def __init__(self, runs_root: str, exp_name: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(runs_root) / exp_name / ts
        self.output_dir = self.run_dir / "output"
        self.checkpoint_dir = self.run_dir / "checkpoints"

        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=False)

    def paths(self) -> RunPaths:
        return RunPaths(self.run_dir, self.output_dir, self.checkpoint_dir)

    def save_config(self, cfg: dict) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    def save_git_state(self) -> None:
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            diff = subprocess.check_output(["git", "diff"]).decode()
            (self.run_dir / "git_sha.txt").write_text(sha)
            (self.run_dir / "git_diff.patch").write_text(diff)
        except Exception:
            pass
