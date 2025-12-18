from pathlib import Path
import argparse
from datetime import datetime

from omegaconf import OmegaConf

from .rnn_trainer import BrainToTextDecoder_Trainer


# resolver para timestamps en YAML: ${now:%Y-%m-%d_%H%M%S}
OmegaConf.register_new_resolver(
    "now",
    lambda fmt="%Y-%m-%d_%H%M%S": datetime.now().strftime(fmt),
    replace=True,
)


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "configs" / "rnn_args.yaml"


def _flatten(maybe_lists):
    out = []
    for x in maybe_lists:
        if x is None:
            continue
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        action="append",
        default=[str(_default_config_path())],
        help="Path to YAML base config. You can pass --config multiple times (they are merged in order).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Path to an override YAML config. Can be specified multiple times.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Dotlist overrides like model.n_units=512. Can be specified multiple times.",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print the final merged config and exit.",
    )

    args = parser.parse_args()

    # Merge base configs provided via --config (in order)
    c
