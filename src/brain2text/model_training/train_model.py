from pathlib import Path
import argparse
from omegaconf import OmegaConf
from .rnn_trainer import BrainToTextDecoder_Trainer


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "configs" / "rnn_args.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=[str(_default_config_path())],
        help="Ruta a YAML. Puedes pasar --config varias veces (base + overrides).",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Imprime el config final y sale.",
    )
    args_cli = parser.parse_args()

    cfgs = [OmegaConf.load(p) for p in args_cli.config]
    cfg = OmegaConf.merge(*cfgs)

    if args_cli.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return

    trainer = BrainToTextDecoder_Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
