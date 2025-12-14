import argparse
from omegaconf import OmegaConf
from brain2text.utils.seed import seed_everything
from brain2text.utils.run_manager import RunManager

# IMPORTANT: this must point to the trainer you will run from src/
from brain2text.training.rnn_ctc_trainer import BrainToTextDecoder_Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)

    seed_everything(int(cfg.get("seed", 0)), deterministic=True)

    exp_name = cfg.get("experiment", {}).get("name", "gru_ctc_baseline")
    runs_root = cfg.get("runs_root", "runs")

    rm = RunManager(runs_root=runs_root, exp_name=exp_name)
    paths = rm.paths()

    rm.save_config(OmegaConf.to_container(cfg, resolve=True))
    rm.save_git_state()

    # Inject canonical run dirs
    cfg["output_dir"] = str(paths.output_dir)
    cfg["checkpoint_dir"] = str(paths.checkpoint_dir)
    cfg["mode"] = "train"

    trainer = BrainToTextDecoder_Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
