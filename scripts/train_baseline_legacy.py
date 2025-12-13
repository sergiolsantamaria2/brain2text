from omegaconf import OmegaConf
from baseline_legacy.model_training.rnn_trainer import BrainToTextDecoder_Trainer

def main():
    args = OmegaConf.load("baseline_legacy/model_training/rnn_args.yaml")
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
