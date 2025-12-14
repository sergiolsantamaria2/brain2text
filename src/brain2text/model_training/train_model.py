from pathlib import Path
import argparse
from omegaconf import OmegaConf
from .rnn_trainer import BrainToTextDecoder_Trainer


def _default_config_path() -> Path:
    # .../repo/src/brain2text/model_training/train_model.py -> repo root = parents[3]
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "configs" / "rnn_args.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(_default_config_path()))
    args_cli = parser.parse_args()

    cfg = OmegaConf.load(args_cli.config)
    trainer = BrainToTextDecoder_Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
