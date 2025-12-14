import argparse
from pathlib import Path
import yaml

from brain2text.utils.seed import seed_everything
from brain2text.utils.run_manager import RunManager
from brain2text.training.trainer import Trainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)  # path to last.ckpt
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    seed_everything(cfg.get("seed", 0))

    run = RunManager(
        base_dir=cfg.get("runs_dir", "runs"),
        run_name=args.run_name or cfg["experiment"]["name"],
        config=cfg,
    )
    run.save_config()

    trainer = Trainer(cfg=cfg, run=run)
    trainer.fit(resume_path=args.resume)
