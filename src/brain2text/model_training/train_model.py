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
    # Permite pasar varios --config y se aplican en orden (el último sobreescribe)
    parser.add_argument(
        "--config",
        action="append",
        default=[str(_default_config_path())],
        help="Config YAML(s). You can pass multiple: --config base.yaml --config exp.yaml",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print final merged config and exit",
    )
    args_cli = parser.parse_args()

    cfgs = [OmegaConf.load(p) for p in args_cli.config]
    cfg = OmegaConf.merge(*cfgs)

    if args_cli.print_config:
        print(OmegaConf.to_yaml(cfg))
        return

    trainer = BrainToTextDecoder_Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
